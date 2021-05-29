__author__ = 'DafniAntotsiou'

'''
This script is heavily based on @openai/baselines.gail.trpo_mpi.
Has been extended to include semi supervised correction network training.
'''

import time
import os
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque

import tensorflow as tf
import numpy as np

from cat_dauggi import tf_util as U
from baselines.common import explained_variance, zipsame, dataset, fmt_row, Dataset
from cat_dauggi import logger
from baselines.common import colorize
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from cat_dauggi.statistics import stats
from baselines.common.mpi_moments import mpi_moments
from cat_dauggi.semi_utils import semi_ob_space, reset_env, semi_loss_func
from cat_dauggi.semi_utils import get_il_prefix
from copy import deepcopy


def traj_segment_generator(pi, env, reward_giver, horizon, stochastic, semi_dataset, semi_loss=False):

    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    full_ob, ob, env = reset_env(semi_dataset, env, random_init=True)

    full_obs = np.array([full_ob for _ in range(horizon)])

    # save success ratio of the trajectories if env returns success
    ep_success = []     # is None if no success in env, else it's boolean
    curr_ep_success = None
    ep_semi_loss = []
    curr_ep_semi_loss = 0
    l2_losses = np.zeros(horizon, 'float32')   # L2 loss per pair
    done = False
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, full_ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets, "full_ob": full_obs,
                   "ep_success": ep_success, "true_rew": true_rews, "ep_semi_loss": ep_semi_loss, "l2_loss": l2_losses}
            ac, vpred = pi.act(stochastic, full_ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            ep_success = []
            ep_semi_loss = []
        i = t % horizon
        obs[i] = ob
        full_obs[i] = full_ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        rew = reward_giver.get_reward(ob, ac)

        ob, true_rew, new, step_dict = env.step(ac)
        rews[i] = rew
        true_rews[i] = true_rew
        # env.render()

        if hasattr(env.env, 'success'):
            curr_ep_success = env.env.success
        elif isinstance(step_dict, dict) and "success" in step_dict:
            curr_ep_success = step_dict["success"]
        # get semi loss for this frame
        l2_loss = semi_loss_func(ac, full_ob, semi_dataset) if semi_dataset else 0
        l2_losses[i] = deepcopy(l2_loss)

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1

        curr_ep_semi_loss += l2_loss

        if new or (semi_dataset is not None and done):
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            ep_success.append(curr_ep_success)
            curr_ep_success = None
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            if semi_loss and semi_dataset is not None:
                ep_semi_loss.append(curr_ep_semi_loss)
                curr_ep_semi_loss = 0
            full_ob, ob, env = reset_env(semi_dataset, env, random_init=True)
            done = False
        else:
            if semi_dataset:
                # is semi network - get the next frame
                full_ob, done = semi_dataset.get_next_frame_full_ob(ob)
            else:
                full_ob = ob

        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, policy_func, reward_giver, expert_dataset, rank,
          pretrained, pretrained_weight, *,
          g_step, d_step, entcoeff, save_per_iter,
          ckpt_dir, log_dir, timesteps_per_batch, task_name,
          gamma, lam,
          max_kl, cg_iters, cg_damping=1e-2,
          vf_stepsize=3e-4, d_stepsize=3e-4, vf_iters=3,
          max_timesteps=0, max_episodes=0, max_iters=0,
          vf_batchsize=128,
          callback=None,
          freeze_g=False,
          freeze_d=False,
          semi_dataset=None,
          semi_loss=False
          ):

    semi_loss = semi_loss and semi_dataset is not None
    l2_w = 0.1

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)

    if rank == 0:
        writer = U.file_writer(log_dir)

        # print all the hyperparameters in the log...
        log_dict = {
            "expert trajectories": expert_dataset.num_traj,
            "algo": "trpo",
            "threads": nworkers,
            "timesteps_per_batch": timesteps_per_batch,
            "timesteps_per_thread": -(-timesteps_per_batch // nworkers),
            "entcoeff": entcoeff,
            "vf_iters": vf_iters,
            "vf_batchsize": vf_batchsize,
            "vf_stepsize": vf_stepsize,
            "d_stepsize": d_stepsize,
            "g_step": g_step,
            "d_step": d_step,
            "max_kl": max_kl,
            "gamma": gamma,
            "lam": lam,
            "l2_weight": l2_w
        }

        if semi_dataset is not None:
            log_dict["semi trajectories"] = semi_dataset.num_traj
        if hasattr(semi_dataset, 'info'):
            log_dict["semi_dataset_info"] = semi_dataset.info

        # print them all together for csv
        logger.log(",".join([str(elem) for elem in log_dict]))
        logger.log(",".join([str(elem) for elem in log_dict.values()]))

        # also print them separately for easy reading:
        for elem in log_dict:
            logger.log(str(elem) + ": " + str(log_dict[elem]))

    # divide the timesteps to the threads
    timesteps_per_batch = -(-timesteps_per_batch // nworkers)       # get ceil of division

    # Setup losses and stuff
    # ----------------------------------------
    if semi_dataset:
        ob_space = semi_ob_space(env, semi_size=semi_dataset.semi_size)
    else:
        ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space=ob_space, ac_space=ac_space, reuse=(pretrained_weight is not None))
    oldpi = policy_func("oldpi", ob_space=ob_space, ac_space=ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    vf_losses = [vferr]
    vf_loss_names = ["vf_loss"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vf")]
    assert len(var_list) == len(vf_var_list) + 1

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_vf_losses = U.function([ob, ac, atarg, ret], losses + vf_losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], vf_losses + [U.flatgrad(vferr, vf_var_list)])

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=40)
    success_buffer = deque(maxlen=40)
    l2_rewbuffer = deque(maxlen=40) if semi_loss and semi_dataset is not None else None
    total_rewbuffer = deque(maxlen=40) if semi_loss and semi_dataset is not None else None

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    not_update = 1 if not freeze_d else 0  # do not update G before D the first time
    # if provide pretrained weight
    loaded = False
    if not U.load_checkpoint_variables(pretrained_weight):
        if U.load_checkpoint_variables(pretrained_weight, check_prefix=get_il_prefix()):
            if rank == 0:
                logger.log("loaded checkpoint variables from " + pretrained_weight)
            loaded = True
    else:
        loaded = True

    if loaded:
        not_update = 0 if any([x.op.name.find("adversary") != -1 for x in U.ALREADY_INITIALIZED]) else 1
        if pretrained_weight and pretrained_weight.rfind("iter_") and \
                pretrained_weight[pretrained_weight.rfind("iter_") + len("iter_"):].isdigit():
            curr_iter = int(pretrained_weight[pretrained_weight.rfind("iter_") + len("iter_"):]) + 1
            print("loaded checkpoint at iteration: " + str(curr_iter))
            iters_so_far = curr_iter
            timesteps_so_far = iters_so_far * timesteps_per_batch

    d_adam = MpiAdam(reward_giver.get_trainable_variables())
    vfadam = MpiAdam(vf_var_list)

    U.initialize()

    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    d_adam.sync()
    vfadam.sync()
    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, reward_giver, timesteps_per_batch, stochastic=True,
                                     semi_dataset=semi_dataset, semi_loss=semi_loss)  # ADD L2 loss to semi trajectories

    g_loss_stats = stats(loss_names + vf_loss_names)
    d_loss_stats = stats(reward_giver.loss_name)
    ep_names = ["True_rewards", "Rewards", "Episode_length", "Success"]
    if semi_loss and semi_dataset is not None:
        ep_names.append("L2_loss")
        ep_names.append("total_rewards")
    ep_stats = stats(ep_names)

    if rank == 0:
        start_time = time.time()
        ch_count = 0
        env_type = env.env.env.__class__.__name__

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # Save model
        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            fname = os.path.join(ckpt_dir, task_name)
            if env_type.find("Pendulum") != -1 or save_per_iter != 1:   # allow pendulum to save all iterations
                fname = os.path.join(ckpt_dir, 'iter_' + str(iters_so_far), 'iter_' + str(iters_so_far))
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver = tf.train.Saver()
            saver.save(tf.get_default_session(), fname, write_meta_graph=False)

        if rank == 0 and time.time() - start_time >= 3600 * ch_count:   # save a different checkpoint every hour
            fname = os.path.join(ckpt_dir, 'hour' + str(ch_count).zfill(3))
            fname = os.path.join(fname, 'iter_' + str(iters_so_far))
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver = tf.train.Saver()
            saver.save(tf.get_default_session(), fname, write_meta_graph=False)
            ch_count += 1


        logger.log("********** Iteration %i ************" % iters_so_far)

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p
        # ------------------ Update G ------------------
        logger.log("Optimizing Policy...")
        for curr_step in range(g_step):
            with timed("sampling"):
                seg = seg_gen.__next__()

            seg["rew"] = seg["rew"] - seg["l2_loss"] * l2_w

            add_vtarg_and_adv(seg, gamma, lam)
            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret, full_ob = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["full_ob"]
            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
            d = Dataset(dict(ob=full_ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)

            if not_update:
                break   # stop G from updating

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(full_ob)     # update running mean/std for policy

            args = seg["full_ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]

            assign_old_eq_new()  # set old parameter values to new parameter values
            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / max_kl)
                # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                    if rank == 0:
                        print("Generator entropy " + str(meanlosses[4]) + ", loss " + str(meanlosses[2]))
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > max_kl * 1.5:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    set_from_flat(thbefore)
                if nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
            with timed("vf"):
                logger.log(fmt_row(13, vf_loss_names))
                for _ in range(vf_iters):
                    vf_b_losses = []
                    for batch in d.iterate_once(vf_batchsize):
                        mbob = batch["ob"]
                        mbret = batch["vtarg"]

                        if hasattr(pi, "ob_rms"):
                            pi.ob_rms.update(mbob)  # update running mean/std for policy
                        *newlosses, g = compute_vflossandgrad(mbob, mbret)
                        g = allmean(g)
                        newlosses = allmean(np.array(newlosses))

                        vfadam.update(g, vf_stepsize)
                        vf_b_losses.append(newlosses)
                    logger.log(fmt_row(13, np.mean(vf_b_losses, axis=0)))

            logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(vf_batchsize):
                newlosses = compute_vf_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"])
                losses.append(newlosses)
            meanlosses,_,_ = mpi_moments(losses, axis=0)

            #########################
            '''
            For evaluation during training.
            Can be commented out for faster training...
            '''
            for ob_batch, ac_batch, full_ob_batch in dataset.iterbatches((ob, ac, full_ob),
                                                          include_final_partial_batch=False,
                                                          batch_size=len(ob)):
                ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
                exp_rew = 0
                for obs, acs in zip(ob_expert, ac_expert):
                    exp_rew += 1 - np.exp(-reward_giver.get_reward(obs, acs)[0][0])
                mean_exp_rew = exp_rew / len(ob_expert)

                gen_rew = 0
                for obs, acs, full_obs in zip(ob_batch, ac_batch, full_ob_batch):
                    gen_rew += 1 - np.exp(-reward_giver.get_reward(obs, acs)[0][0])
                mean_gen_rew = gen_rew / len(ob_batch)
                if rank == 0:
                    logger.log(
                        "Generator step " + str(curr_step) + ": Dicriminator reward of expert traj "
                        + str(mean_exp_rew) + " vs gen traj " + str(mean_gen_rew))
            #########################

        if not not_update:
            g_losses = meanlosses
            for (lossname, lossval) in zip(loss_names + vf_loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)
            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        # ------------------ Update D ------------------
        if not freeze_d:
            logger.log("Optimizing Discriminator...")
            batch_size = len(ob) // d_step
            d_losses = []  # list of tuples, each of which gives the loss for a minibatch
            for ob_batch, ac_batch, full_ob_batch in dataset.iterbatches((ob, ac, full_ob),
                                                          include_final_partial_batch=False,
                                                          batch_size=batch_size):
                ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
                #########################
                '''
                For evaluation during training.
                Can be commented out for faster training...
                '''
                exp_rew = 0
                for obs, acs in zip(ob_expert, ac_expert):
                    exp_rew += 1 - np.exp(-reward_giver.get_reward(obs, acs)[0][0])
                mean_exp_rew = exp_rew / len(ob_expert)

                gen_rew = 0

                for obs, acs in zip(ob_batch, ac_batch):
                    gen_rew += 1 - np.exp(-reward_giver.get_reward(obs, acs)[0][0])

                mean_gen_rew = gen_rew / len(ob_batch)
                if rank == 0:
                    logger.log("Dicriminator reward of expert traj " + str(mean_exp_rew) +
                               " vs gen traj " + str(mean_gen_rew))
                #########################
                # update running mean/std for reward_giver
                if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
                loss_input = (ob_batch, ac_batch, ob_expert, ac_expert)
                *newlosses, g = reward_giver.lossandgrad(*loss_input)
                d_adam.update(allmean(g), d_stepsize)
                d_losses.append(newlosses)
            logger.log(fmt_row(13, reward_giver.loss_name))
            logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"], seg["ep_success"], seg["ep_semi_loss"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews, true_rets, success, semi_losses = map(flatten_lists, zip(*listoflrpairs))

        # success
        success = [float(elem) for elem in success if isinstance(elem, (int, float, bool))]  # remove potential None types
        if not success:
            success = [-1]  # set success to -1 if env has no success flag
        success_buffer.extend(success)

        true_rewbuffer.extend(true_rets)
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        if semi_loss and semi_dataset is not None:
            semi_losses = [elem * l2_w for elem in semi_losses]
            total_rewards = rews
            total_rewards = [re_elem - l2_elem for re_elem, l2_elem in zip(total_rewards, semi_losses)]
            l2_rewbuffer.extend(semi_losses)
            total_rewbuffer.extend(total_rewards)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("EpSuccess", np.mean(success_buffer))

        if semi_loss and semi_dataset is not None:
            logger.record_tabular("EpSemiLoss", np.mean(l2_rewbuffer))
            logger.record_tabular("EpTotalReward", np.mean(total_rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("ItersSoFar", iters_so_far)

        if rank == 0:
            logger.dump_tabular()
            if not not_update:
                g_loss_stats.add_all_summary(writer, g_losses, iters_so_far)
            if not freeze_d:
                d_loss_stats.add_all_summary(writer, np.mean(d_losses, axis=0), iters_so_far)

            # default buffers
            ep_buffers = [np.mean(true_rewbuffer), np.mean(rewbuffer), np.mean(lenbuffer), np.mean(success_buffer)]

            if semi_loss and semi_dataset is not None:
                ep_buffers.append(np.mean(l2_rewbuffer))
                ep_buffers.append(np.mean(total_rewbuffer))

            ep_stats.add_all_summary(writer, ep_buffers, iters_so_far)

        if not_update and not freeze_g:
            not_update -= 1


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
