__author__ = 'DafniAntotsiou'

'''
This script is heavily based on @openai/baselines.gail.trpo_mpi
Has been extended to use a semi supervised correction network
as expert teacher for the imitation agent.
'''

import time
import os
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque
from collections import OrderedDict

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
from cat_dauggi.semi_utils import semi_ob_space, reset_env, get_il_prefix, get_semi_prefix, semi_loss_func
from cat_dauggi.diversity import mean_trajectory


def traj_segment_generator(pi, env, reward_giver, horizon, stochastic, semi_dataset, semi_loss=False,
                           reward_threshold=None, sparse_reward=False):

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

    full_ob, ob, env = reset_env(semi_dataset, env, random_init=False)

    full_obs = np.array([full_ob for _ in range(horizon)])

    # save success ratio of the trajectories if env returns success
    ep_success = []     # is None if no success in env, else it's boolean
    curr_ep_success = None
    ep_semi_loss = []
    curr_ep_semi_loss = 0
    l2_losses = np.zeros(horizon, 'float32')   # L2 loss per pair
    remove_trajectory = False
    done = False
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, full_ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0 and not remove_trajectory:
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

        if hasattr(reward_giver, '_labels'):
            rew = reward_giver.get_reward(ob, ac, get_semi_prefix() if semi_dataset else get_il_prefix())
        else:
            rew = reward_giver.get_reward(ob, ac)

        ob, true_rew, new, step_dict = env.step(ac)
        # env.render()
        rews[i] = rew
        true_rews[i] = true_rew

        if hasattr(env.env.env, 'success'):
            curr_ep_success = env.env.env.success
        elif isinstance(step_dict, dict) and "success" in step_dict:
            curr_ep_success = step_dict["success"] if curr_ep_success is None \
                else step_dict["success"]   # or curr_ep_success

        if curr_ep_success is None and sparse_reward:
            logger.log("No success in the environment, cannot filter by sparse reward. Terminating...")
            exit(1)

        # get semi loss for this frame
        l2_loss = semi_loss_func(ac, full_ob, semi_dataset) if semi_dataset else 0
        l2_losses[i] = l2_loss

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1

        curr_ep_semi_loss += l2_loss

        remove_trajectory = False
        if new or (semi_dataset is not None and done):
            if (not sparse_reward and reward_threshold and cur_ep_true_ret < reward_threshold) \
                    or (sparse_reward and not curr_ep_success):
                msg = "Removing trajectory with reward score " + str(cur_ep_true_ret)
                if sparse_reward:
                    msg += " and sparse reward = " + str(curr_ep_success)
                logger.log(msg)
                remove_trajectory = True

            if remove_trajectory:
                # current trajectory is not good enough, remove it from the buffers
                t = max(t - cur_ep_len, int(t/horizon)*horizon-1)  # make sure to reset to 0 if there is a leftover trajectory
            else:
                ep_rets.append(cur_ep_ret)
                ep_true_rets.append(cur_ep_true_ret)
                ep_lens.append(cur_ep_len)
                ep_success.append(curr_ep_success)

                if semi_loss and semi_dataset is not None:
                    ep_semi_loss.append(curr_ep_semi_loss)

            curr_ep_success = None
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            curr_ep_semi_loss = 0
            full_ob, ob, env = reset_env(semi_dataset, env, random_init=False)
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


def learn(env, policy_func, reward_giver,
          semi_dataset, rank,
          pretrained_weight, *,
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
          pretrained_il=None,
          pretrained_semi=None,
          semi_loss=False,
          expert_reward_threshold=None,     # filter experts based on reward
          expert_label=get_semi_prefix(),
          sparse_reward=False               # filter experts based on success flag (sparse reward)
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
            # "expert trajectories": expert_dataset.num_traj,
            "expert model": pretrained_semi,
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
        }

        if semi_dataset is not None:
            log_dict["semi trajectories"] = semi_dataset.num_traj
        if hasattr(semi_dataset, 'info'):
            log_dict["semi_dataset_info"] = semi_dataset.info
        if expert_reward_threshold is not None:
            log_dict["expert reward threshold"] = expert_reward_threshold
        log_dict["sparse reward"] = sparse_reward

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
    ob_space = OrderedDict([(label, env[label].observation_space) for label in env])

    if semi_dataset and get_semi_prefix() in env:   # semi ob space is different
        semi_obs_space = semi_ob_space(env[get_semi_prefix()], semi_size=semi_dataset.semi_size)
        ob_space[get_semi_prefix()] = semi_obs_space
    else:
        print("no semi dataset")
        # raise RuntimeError

    vf_stepsize = {label: vf_stepsize for label in env}

    ac_space = {label: env[label].action_space for label in ob_space}
    pi = {label: policy_func("pi", ob_space=ob_space[label], ac_space=ac_space[label], prefix=label) for label in ob_space}
    oldpi = {label: policy_func("oldpi", ob_space=ob_space[label], ac_space=ac_space[label], prefix=label) for label in ob_space}
    atarg = {label: tf.placeholder(dtype=tf.float32, shape=[None]) for label in ob_space}  # Target advantage function (if applicable)
    ret = {label: tf.placeholder(dtype=tf.float32, shape=[None]) for label in ob_space}  # Empirical return

    ob = {label: U.get_placeholder_cached(name=label+"ob") for label in ob_space}
    ac = {label: pi[label].pdtype.sample_placeholder([None]) for label in ob_space}

    kloldnew = {label: oldpi[label].pd.kl(pi[label].pd) for label in ob_space}
    ent = {label: pi[label].pd.entropy() for label in ob_space}
    meankl = {label: tf.reduce_mean(kloldnew[label]) for label in ob_space}
    meanent = {label: tf.reduce_mean(ent[label]) for label in ob_space}
    entbonus = {label: entcoeff * meanent[label] for label in ob_space}

    vferr = {label: tf.reduce_mean(tf.square(pi[label].vpred - ret[label])) for label in ob_space}

    ratio = {label: tf.exp(pi[label].pd.logp(ac[label]) - oldpi[label].pd.logp(ac[label])) for label in ob_space}  # advantage * pnew / pold
    surrgain = {label: tf.reduce_mean(ratio[label] * atarg[label]) for label in ob_space}

    optimgain = {label: surrgain[label] + entbonus[label] for label in ob_space}
    losses = {label: [optimgain[label], meankl[label], entbonus[label], surrgain[label], meanent[label]] for label in ob_space}
    loss_names = {label: [label + name for name in ["optimgain", "meankl", "entloss", "surrgain", "entropy"]] for label in ob_space}

    vf_losses = {label: [vferr[label]] for label in ob_space}
    vf_loss_names = {label: [label + "vf_loss"] for label in ob_space}

    dist = {label: meankl[label] for label in ob_space}

    all_var_list = {label: pi[label].get_trainable_variables() for label in ob_space}
    var_list = {label: [v for v in all_var_list[label] if "pol" in v.name or "logstd" in v.name] for label in ob_space}
    vf_var_list = {label: [v for v in all_var_list[label] if "vf" in v.name] for label in ob_space}
    for label in ob_space:
        assert len(var_list[label]) == len(vf_var_list[label]) + 1

    get_flat = {label: U.GetFlat(var_list[label]) for label in ob_space}
    set_from_flat = {label: U.SetFromFlat(var_list[label]) for label in ob_space}
    klgrads = {label: tf.gradients(dist[label], var_list[label]) for label in ob_space}
    flat_tangent = {label: tf.placeholder(dtype=tf.float32, shape=[None], name=label+"flat_tan") for label in ob_space}
    fvp = {}
    for label in ob_space:
        shapes = [var.get_shape().as_list() for var in var_list[label]]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[label][start:start+sz], shape))
            start += sz
        gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads[label], tangents)])  # pylint: disable=E1111
        fvp[label] = U.flatgrad(gvp, var_list[label])

    assign_old_eq_new = {label: U.function([], [], updates=[tf.assign(oldv, newv)
                         for (oldv, newv) in zipsame(oldpi[label].get_variables(), pi[label].get_variables())])
                         for label in ob_space}
    compute_losses = {label: U.function([ob[label], ac[label], atarg[label]], losses[label]) for label in ob_space}

    compute_vf_losses = {label: U.function([ob[label], ac[label], atarg[label], ret[label]],
                                            losses[label] + vf_losses[label]) for label in ob_space}

    compute_lossandgrad = {label: U.function([ob[label], ac[label], atarg[label]], losses[label] +
                                             [U.flatgrad(optimgain[label], var_list[label])]) for label in ob_space}
    compute_fvp = {label: U.function([flat_tangent[label], ob[label], ac[label], atarg[label]], fvp[label])
                   for label in ob_space}

    compute_vflossandgrad = {label: U.function([ob[label], ret[label]], vf_losses[label] +
                                               [U.flatgrad(vferr[label], vf_var_list[label])]) for label in ob_space}

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

    episodes_so_far = {label: 0 for label in ob_space}
    timesteps_so_far = {label: 0 for label in ob_space}
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = {label: deque(maxlen=40) for label in ob_space}  # rolling buffer for episode lengths
    rewbuffer = {label: deque(maxlen=40) for label in ob_space}  # rolling buffer for episode rewards
    true_rewbuffer = {label: deque(maxlen=40) for label in ob_space}
    success_buffer = {label: deque(maxlen=40) for label in ob_space}
    # L2 only for semi network
    l2_rewbuffer = deque(maxlen=40) if semi_loss and semi_dataset is not None else None
    total_rewbuffer = deque(maxlen=40) if semi_loss and semi_dataset is not None else None

    not_update = 1 if not freeze_d else 0  # do not update G before D the first time
    loaded = False
    # if provide pretrained weight
    if not U.load_checkpoint_variables(pretrained_weight, include_no_prefix_vars=True):
        # if no general checkpoint available, check sub-checkpoints for both networks
        if U.load_checkpoint_variables(pretrained_il, prefix=get_il_prefix(), include_no_prefix_vars=False):
            if rank == 0:
                logger.log("loaded checkpoint variables from " + pretrained_il)
            loaded = True
        elif expert_label == get_il_prefix():
            logger.log("ERROR no available cat_dauggi expert model in ", pretrained_il)
            exit(1)

        if U.load_checkpoint_variables(pretrained_semi, prefix=get_semi_prefix(), include_no_prefix_vars=False):
            if rank == 0:
                logger.log("loaded checkpoint variables from " + pretrained_semi)
            loaded = True
        elif expert_label == get_semi_prefix():
            if rank == 0:
                logger.log("ERROR no available semi expert model in ", pretrained_semi)
            exit(1)
    else:
        loaded = True
        if rank == 0:
            logger.log("loaded checkpoint variables from " + pretrained_weight)

    if loaded:
        not_update = 0 if any([x.op.name.find("adversary") != -1 for x in U.ALREADY_INITIALIZED]) else 1
        if pretrained_weight and pretrained_weight.rfind("iter_") and \
                pretrained_weight[pretrained_weight.rfind("iter_") + len("iter_"):].isdigit():
            curr_iter = int(pretrained_weight[pretrained_weight.rfind("iter_") + len("iter_"):]) + 1

            if rank == 0:
                print("loaded checkpoint at iteration: " + str(curr_iter))
            iters_so_far = curr_iter
            for label in timesteps_so_far:
                timesteps_so_far[label] = iters_so_far * timesteps_per_batch


    d_adam = MpiAdam(reward_giver.get_trainable_variables())
    vfadam = {label: MpiAdam(vf_var_list[label]) for label in ob_space}

    U.initialize()
    d_adam.sync()

    for label in ob_space:
        th_init = get_flat[label]()
        MPI.COMM_WORLD.Bcast(th_init, root=0)
        set_from_flat[label](th_init)
        vfadam[label].sync()
        if rank == 0:
            print(label + "Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = {label: traj_segment_generator(pi[label], env[label], reward_giver, timesteps_per_batch, stochastic=True,
                                             semi_dataset=semi_dataset if label == get_semi_prefix() else None, semi_loss=semi_loss,
                                             reward_threshold=expert_reward_threshold if label == expert_label
                                             else None, sparse_reward=sparse_reward if label == expert_label
                                             else False)
               for label in ob_space}

    g_losses = {}

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    g_loss_stats = {label: stats(loss_names[label] + vf_loss_names[label]) for label in ob_space if label != expert_label}
    d_loss_stats = stats(reward_giver.loss_name)
    ep_names = ["True_rewards", "Rewards", "Episode_length", "Success"]

    ep_stats = {label: None for label in ob_space}
    # cat_dauggi network stats
    if get_il_prefix() in ep_stats:
        ep_stats[get_il_prefix()] = stats([name for name in ep_names])

    # semi network stats
    if get_semi_prefix() in ep_stats:
        if semi_loss and semi_dataset is not None:
            ep_names.append("L2_loss")
            ep_names.append("total_rewards")
        ep_stats[get_semi_prefix()] = stats([get_semi_prefix() + name for name in ep_names])

    if rank == 0:
        start_time = time.time()
        ch_count = 0
        env_type = env[expert_label].env.env.__class__.__name__

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and any([timesteps_so_far[label] >= max_timesteps for label in ob_space]):
            break
        elif max_episodes and any([episodes_so_far[label] >= max_episodes for label in ob_space]):
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # Save model
        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            fname = os.path.join(ckpt_dir, task_name)
            if env_type.find("Pendulum") != -1 or save_per_iter != 1:
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

        def fisher_func_builder(label):
            def fisher_vector_product(p):
                return allmean(compute_fvp[label](p, *fvpargs)) + cg_damping * p
            return fisher_vector_product

        # ------------------ Update G ------------------
        d = {label: None for label in ob_space}
        segs = {label: None for label in ob_space}
        logger.log("Optimizing Policy...")
        for curr_step in range(g_step):
            for label in ob_space:

                if curr_step and label == expert_label:  # get expert trajectories only for one g_step which is same as d_step
                    continue

                logger.log("Optimizing Policy " + label + "...")
                with timed("sampling"):
                    segs[label] = seg = seg_gen[label].__next__()

                seg["rew"] = seg["rew"] - seg["l2_loss"] * l2_w

                add_vtarg_and_adv(seg, gamma, lam)
                ob, ac, atarg, tdlamret, full_ob = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["full_ob"]
                vpredbefore = seg["vpred"]  # predicted value function before udpate
                atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
                d[label] = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)

                if not_update or label == expert_label:
                    continue   # stop G from updating

                if hasattr(pi[label], "ob_rms"): pi[label].ob_rms.update(full_ob)  # update running mean/std for policy

                args = seg["full_ob"], seg["ac"], atarg
                fvpargs = [arr[::5] for arr in args]

                assign_old_eq_new[label]()  # set old parameter values to new parameter values
                with timed("computegrad"):
                    *lossbefore, g = compute_lossandgrad[label](*args)
                lossbefore = allmean(np.array(lossbefore))
                g = allmean(g)
                if np.allclose(g, 0):
                    logger.log("Got zero gradient. not updating")
                else:
                    with timed("cg"):
                        stepdir = cg(fisher_func_builder(label), g, cg_iters=cg_iters, verbose=rank == 0)
                    assert np.isfinite(stepdir).all()
                    shs = .5*stepdir.dot(fisher_func_builder(label)(stepdir))
                    lm = np.sqrt(shs / max_kl)
                    # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                    fullstep = stepdir / lm
                    expectedimprove = g.dot(fullstep)
                    surrbefore = lossbefore[0]
                    stepsize = 1.0
                    thbefore = get_flat[label]()
                    for _ in range(10):
                        thnew = thbefore + fullstep * stepsize
                        set_from_flat[label](thnew)
                        meanlosses = surr, kl, *_ = allmean(np.array(compute_losses[label](*args)))
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
                        set_from_flat[label](thbefore)
                    if nworkers > 1 and iters_so_far % 20 == 0:
                        paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam[label].getflat().sum()))  # list of tuples
                        assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

            expert_dataset = d[expert_label]

            if not_update:
                break

            for label in ob_space:
                if label == expert_label:
                    continue

                with timed("vf"):
                    logger.log(fmt_row(13, vf_loss_names[label]))
                    for _ in range(vf_iters):
                        vf_b_losses = []
                        for batch in d[label].iterate_once(vf_batchsize):
                            mbob = batch["ob"]
                            mbret = batch["vtarg"]
                            *newlosses, g = compute_vflossandgrad[label](mbob, mbret)
                            g = allmean(g)
                            newlosses = allmean(np.array(newlosses))

                            vfadam[label].update(g, vf_stepsize[label])
                            vf_b_losses.append(newlosses)
                        logger.log(fmt_row(13, np.mean(vf_b_losses, axis=0)))

                    logger.log("Evaluating losses...")
                    losses = []
                    for batch in d[label].iterate_once(vf_batchsize):
                        newlosses = compute_vf_losses[label](batch["ob"], batch["ac"], batch["atarg"],
                                                                 batch["vtarg"])
                        losses.append(newlosses)
                    g_losses[label], _, _ = mpi_moments(losses, axis=0)

                #########################
                for ob_batch, ac_batch, full_ob_batch in dataset.iterbatches((segs[label]["ob"], segs[label]["ac"],
                                                                              segs[label]["full_ob"]),
                                                                             include_final_partial_batch=False,
                                                                             batch_size=len(ob)):
                    expert_batch = expert_dataset.next_batch(len(ob))

                    ob_expert, ac_expert = expert_batch["ob"], expert_batch["ac"]

                    exp_rew = 0
                    exp_rews = None
                    for obs, acs in zip(ob_expert, ac_expert):
                        curr_rew = reward_giver.get_reward(obs, acs)[0][0] \
                                   if not hasattr(reward_giver, '_labels') else \
                                   reward_giver.get_reward(obs, acs, label)
                        if isinstance(curr_rew, tuple):
                            curr_rew, curr_rews = curr_rew
                            exp_rews = 1 - np.exp(-curr_rews) if exp_rews is None else exp_rews + 1 - np.exp(-curr_rews)
                        exp_rew += 1 - np.exp(-curr_rew)
                    mean_exp_rew = exp_rew / len(ob_expert)
                    mean_exp_rews = exp_rews / len(ob_expert) if exp_rews is not None else None

                    gen_rew = 0
                    gen_rews = None
                    for obs, acs, full_obs in zip(ob_batch, ac_batch, full_ob_batch):
                        curr_rew = reward_giver.get_reward(obs, acs)[0][0] \
                                   if not hasattr(reward_giver, '_labels') else \
                                   reward_giver.get_reward(obs, acs, label)
                        if isinstance(curr_rew, tuple):
                            curr_rew, curr_rews = curr_rew
                            gen_rews = 1 - np.exp(-curr_rews) if gen_rews is None else gen_rews + 1 - np.exp(-curr_rews)
                        gen_rew += 1 - np.exp(-curr_rew)
                    mean_gen_rew = gen_rew / len(ob_batch)
                    mean_gen_rews = gen_rews / len(ob_batch) if gen_rews is not None else None
                    if rank == 0:
                        msg = "Network " + label + \
                            " Generator step " + str(curr_step) + ": Dicriminator reward of expert traj " \
                            + str(mean_exp_rew) + " vs gen traj " + str(mean_gen_rew)
                        if mean_exp_rews is not None and mean_gen_rews is not None:
                            msg += "\nDiscriminator multi rewards of expert " + str(mean_exp_rews) + " vs gen " \
                                    + str(mean_gen_rews)
                        logger.log(msg)
                #########################

        if not not_update:
            for label in g_losses:
                for (lossname, lossval) in zip(loss_names[label] + vf_loss_names[label], g_losses[label]):
                    logger.record_tabular(lossname, lossval)
                logger.record_tabular(label + "ev_tdlam_before", explained_variance(segs[label]["vpred"],
                                                                                    segs[label]["tdlamret"]))

        # ------------------ Update D ------------------
        if not freeze_d:
            logger.log("Optimizing Discriminator...")
            batch_size = len(list(segs.values())[0]['ob']) // d_step
            expert_dataset = d[expert_label]
            batch_gen = {label: dataset.iterbatches((segs[label]["ob"], segs[label]["ac"]),
                                                    include_final_partial_batch=False,
                                                    batch_size=batch_size) for label in segs if label != expert_label}

            d_losses = []  # list of tuples, each of which gives the loss for a minibatch
            for step in range(d_step):
                g_ob = {}
                g_ac = {}
                for label in batch_gen:   # get batches for different gens
                    g_ob[label], g_ac[label] = batch_gen[label].__next__()

                expert_batch = expert_dataset.next_batch(batch_size)

                ob_expert, ac_expert = expert_batch["ob"], expert_batch["ac"]

                for label in g_ob:
                    #########################
                    exp_rew = 0
                    exp_rews = None
                    for obs, acs in zip(ob_expert, ac_expert):
                        curr_rew = reward_giver.get_reward(obs, acs)[0][0] \
                            if not hasattr(reward_giver, '_labels') else \
                            reward_giver.get_reward(obs, acs, label)
                        if isinstance(curr_rew, tuple):
                            curr_rew, curr_rews = curr_rew
                            exp_rews = 1 - np.exp(-curr_rews) if exp_rews is None else exp_rews + 1 - np.exp(-curr_rews)
                        exp_rew += 1 - np.exp(-curr_rew)
                    mean_exp_rew = exp_rew / len(ob_expert)
                    mean_exp_rews = exp_rews / len(ob_expert) if exp_rews is not None else None

                    gen_rew = 0
                    gen_rews = None
                    for obs, acs in zip(g_ob[label], g_ac[label]):
                        curr_rew = reward_giver.get_reward(obs, acs)[0][0] \
                            if not hasattr(reward_giver, '_labels') else \
                            reward_giver.get_reward(obs, acs, label)
                        if isinstance(curr_rew, tuple):
                            curr_rew, curr_rews = curr_rew
                            gen_rews = 1 - np.exp(-curr_rews) if gen_rews is None else gen_rews + 1 - np.exp(-curr_rews)
                        gen_rew += 1 - np.exp(-curr_rew)
                    mean_gen_rew = gen_rew / len(g_ob[label])
                    mean_gen_rews = gen_rews / len(g_ob[label]) if gen_rews is not None else None
                    if rank == 0:
                        msg = "Dicriminator reward of expert traj " + str(mean_exp_rew) + " vs " + label + \
                            "gen traj " + str(mean_gen_rew)
                        if mean_exp_rews is not None and mean_gen_rews is not None:
                            msg += "\nDiscriminator multi expert rewards " + str(mean_exp_rews) + " vs " + label + \
                                   "gen " + str(mean_gen_rews)
                        logger.log(msg)
                        #########################

                # update running mean/std for reward_giver
                if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(
                    np.concatenate(list(g_ob.values()) + [ob_expert], 0))
                *newlosses, g = reward_giver.lossandgrad(
                    *(list(g_ob.values()) + list(g_ac.values()) + [ob_expert] + [ac_expert]))
                d_adam.update(allmean(g), d_stepsize)
                d_losses.append(newlosses)
                logger.log(fmt_row(13, reward_giver.loss_name))
                logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        for label in ob_space:
            lrlocal = (segs[label]["ep_lens"], segs[label]["ep_rets"], segs[label]["ep_true_rets"],
                       segs[label]["ep_success"], segs[label]["ep_semi_loss"])  # local values

            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
            lens, rews, true_rets, success, semi_losses = map(flatten_lists, zip(*listoflrpairs))

            # success
            success = [float(elem) for elem in success if isinstance(elem, (int, float, bool))]  # remove potential None types
            if not success:
                success = [-1]  # set success to -1 if env has no success flag
            success_buffer[label].extend(success)

            true_rewbuffer[label].extend(true_rets)
            lenbuffer[label].extend(lens)
            rewbuffer[label].extend(rews)

            if semi_loss and semi_dataset is not None and label == get_semi_prefix():
                semi_losses = [elem * l2_w for elem in semi_losses]
                total_rewards = rews
                total_rewards = [re_elem - l2_elem for re_elem, l2_elem in zip(total_rewards, semi_losses)]
                l2_rewbuffer.extend(semi_losses)
                total_rewbuffer.extend(total_rewards)

            logger.record_tabular(label + "EpLenMean", np.mean(lenbuffer[label]))
            logger.record_tabular(label + "EpRewMean", np.mean(rewbuffer[label]))
            logger.record_tabular(label + "EpTrueRewMean", np.mean(true_rewbuffer[label]))
            logger.record_tabular(label + "EpSuccess", np.mean(success_buffer[label]))

            if semi_loss and semi_dataset is not None and label == get_semi_prefix():
                logger.record_tabular(label + "EpSemiLoss", np.mean(l2_rewbuffer))
                logger.record_tabular(label + "EpTotalLoss", np.mean(total_rewbuffer))
            logger.record_tabular(label + "EpThisIter", len(lens))
            episodes_so_far[label] += len(lens)
            timesteps_so_far[label] += sum(lens)

            logger.record_tabular(label + "EpisodesSoFar", episodes_so_far[label])
            logger.record_tabular(label + "TimestepsSoFar", timesteps_so_far[label])
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        iters_so_far += 1
        logger.record_tabular("ItersSoFar", iters_so_far)


        if rank == 0:
            logger.dump_tabular()
            if not not_update:
                for label in g_loss_stats:
                    g_loss_stats[label].add_all_summary(writer, g_losses[label], iters_so_far)
            if not freeze_d:
                d_loss_stats.add_all_summary(writer, np.mean(d_losses, axis=0), iters_so_far)

            for label in ob_space:
                # default buffers
                ep_buffers = [np.mean(true_rewbuffer[label]), np.mean(rewbuffer[label]), np.mean(lenbuffer[label]),
                              np.mean(success_buffer[label])]

                if semi_loss and semi_dataset is not None and label == get_semi_prefix():
                    ep_buffers.append(np.mean(l2_rewbuffer))
                    ep_buffers.append(np.mean(total_rewbuffer))

                ep_stats[label].add_all_summary(writer, ep_buffers, iters_so_far)

        if not_update and not freeze_g:
            not_update -= 1


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
