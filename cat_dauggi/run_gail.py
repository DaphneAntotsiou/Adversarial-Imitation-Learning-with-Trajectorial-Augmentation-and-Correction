__author__ = 'DafniAntotsiou'

"""
This script is heavily based on @openai/baselines.gail.run_mujoco
"""

import argparse
import os.path as osp
import logging
from mpi4py import MPI
from tqdm import tqdm
import numpy as np
import gym
from cat_dauggi.mlp_policy_ext import MlpPolicy as mlp_policy
from baselines.common import set_global_seeds
from baselines.common.misc_util import boolean_flag
from baselines import bench
from cat_dauggi import logger
from cat_dauggi.mujoco_dset import Mujoco_Dset
import tensorflow as tf
from copy import deepcopy
from cat_dauggi import tf_util as U
from cat_dauggi.semi_utils import get_il_prefix, get_semi_prefix, semi_ob_space, reset_env, reset_envs
from cat_dauggi.eval_utils import evaluate_runs
try:
    import gym_ext
    import adroit_hand_ext
except ImportError:
    pass

CUSTOM_TIMESTEP = False


def none_or_str(value):
    if value == 'None':
        return None
    return value


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='HalfCheetah_ext-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--expert_path', type=str, default='data/HalfCheetah/trajectories.npz')
    # semi supervised CAT network
    boolean_flag(parser, 'semi', default=False, help='run semi-supervised network')
    parser.add_argument('--semi_path', type=str,
                        default="data/HalfCheetah/augmented_trajectories.npz")
    parser.add_argument('--semi_label', type=str, choices=['acs', 'hpe'], default='acs')

    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='gail/')
    parser.add_argument('--log_dir', help='the directory to save log file', default=None)
    parser.add_argument('--load_model_path', help='dir to load the trajectories for evaluation', type=str,
                        default=None)
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='evaluate')
    boolean_flag(parser, 'eval_runs', default=False, help='evaluate all the checkpoints of a train run')
    boolean_flag(parser, 'eval_cat', default=False, help='evaluate all the hours of a train run')
    # for evaluatation n
    boolean_flag(parser, 'stochastic_policy', default=True, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # debugging GAIL...
    boolean_flag(parser, 'freeze_g', default=False, help='freeze generator and train discriminator only')
    boolean_flag(parser, 'freeze_d', default=False, help='freeze discriminator and train generator only')
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number o`f steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=64)
    parser.add_argument('--adversary_hidden_size', type=int, default=64)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')

    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Training Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=1)
    parser.add_argument('--num_iters', help='max number of iterations', type=int, default=3e3)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    parser.add_argument('--timesteps_per_batch', help='total generator timesteps per batch', type=int, default=int(2 ** 14))
    parser.add_argument('--continue_checkpoint', type=none_or_str, help='directory of latest checkpoint to continue',
                        default=None)
    parser.add_argument('--policy', type=str, choices=['custom', 'default'], help='policy architecture', default='default')
    boolean_flag(parser, 'render', default=False, help='render trajectories for qualitative evaluation')
    # parser.add_argument('--sigma', type=none_or_str, help='maximum distance of random initialisation', default=None)
    parser.add_argument('--g_lr', type=float, help='generator learning rate', default=1e-3)
    parser.add_argument('--d_lr', type=float, help='discriminator learning rate', default=3e-4)
    parser.add_argument('--batch_size', type=float, help='batch size', default=128)
    parser.add_argument('--optim_epochs', type=float, default=5)
    return parser.parse_args()


def get_task_name(args):
    task_name = args.algo + "_gail."
    if args.semi and args.semi_path:
        task_name += "semi."

    if args.continue_checkpoint:
        task_name += "continue."
    elif args.pretrained:
        task_name += "with_pretrained.BC" + str("%.g." % args.BC_max_iter)

    if args.freeze_d:
        task_name += "G_only."
    elif args.freeze_g:
        task_name += "D_only."

    # debug hyperparameters

    task_name += str("%.e." % args.g_lr)
    task_name += str("%.e." % args.d_lr)
    task_name += str("%g." % args.batch_size)

    if args.traj_limitation != np.inf:
        task_name += "transition_limitation_%d." % args.traj_limitation
    task_name += args.env_id
    task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
        ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    task_name += ".seed_" + str(args.seed)
    return task_name


def main(args):
    graph = tf.Graph()
    sess = U.make_session(graph=graph).__enter__()

    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    # just-in-case print...
    if args.semi and args.semi_path:
        print("IS CAT NETWORK")
    else:
        print("IS GAIL NETWORK")

    env = bench.Monitor(env, None, allow_early_resets=True)
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name) if args.log_dir else osp.join(args.checkpoint_dir, 'log')

    if args.semi:
        if args.semi_path:
            # load semi dataset
            from cat_dauggi.semidset import SemiDset
            semi_dataset = SemiDset(expert_path=args.semi_path, traj_limitation=-1, seed=args.seed,
                                    semi_label=args.semi_label)  # it can be hpe or acs
        else:
            print("No valid path for semi dataset. Terminating...")
            exit(1)
    else:
        semi_dataset = None

    def policy_fn(name, ob_space, ac_space, reuse=False, prefix=''):
        if args.policy == 'default':
            return mlp_policy(name=name, ob_space=ob_space, ac_space=ac_space,
                              hid_size=args.policy_hidden_size, num_hid_layers=2, prefix=prefix)
        else:
            NotImplementedError

    if args.task == 'train':
        dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation, seed=args.seed)

        if args.policy == 'default':
            from cat_dauggi.adversary_ext import TransitionClassifier_Ext as TransitionClassifier
        else:
            raise NotImplementedError

        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
        train(env,
              args.seed,
              policy_fn,
              reward_giver,
              dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              # args.num_timesteps,
              args.num_iters,
              args.timesteps_per_batch,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              args.g_lr,
              args.d_lr,
              args.batch_size,
              args.optim_epochs,
              args.max_kl,
              task_name,
              args.continue_checkpoint,
              semi_dataset=semi_dataset,
              freeze_g=args.freeze_g,
              freeze_d=args.freeze_d,
              )
    elif args.task == 'evaluate':
        dir_separation = True

        func_list = [env, policy_fn]
        func_dict = {'load_model_path': args.load_model_path,
                     'timesteps_per_batch': env.spec.max_episode_steps,
                     'number_trajs': 100,  # if semi_dataset is None else semi_dataset.semi_trajectories * 10,
                     'stochastic_policy': args.stochastic_policy,
                     'save': args.save_sample,
                     'render': args.render,
                     'semi_dataset': semi_dataset,
                     'network_prefix': 'semi' if args.semi else 'il',
                     'add_noise': False}

        keyword = 'hour'
        step = 1
        if args.env_id.find("Pendulum") != -1:
            keyword = "iter_"
            step = 10

        if args.eval_runs:
            func_dict["save"] = False


            eval_csv = evaluate_runs(runner, func_list=func_list, func_dict=func_dict, seed=args.seed, keyword=keyword, step=step,
                                     stochastic=args.stochastic_policy, continue_hour=True, dir_separation=dir_separation)

        if args.eval_cat or not args.eval_runs:
            # evaluation of a single checkpoint
            if args.eval_cat and args.eval_runs:
                from cat_dauggi.eval_utils import best_checkpoint_dir
                # find the best checkpoint from hours csv
                func_dict['load_model_path'] = best_checkpoint_dir(log=eval_csv, dir=args.load_model_path,
                                                                   dir_separation=dir_separation, keyword=keyword)

            if semi_dataset is not None and (args.render or args.eval_cat):
                # add second environment for the original noisy trajectories
                orig_env = gym.make(args.env_id)
                orig_env = bench.Monitor(orig_env, logger.get_dir() and
                                         osp.join(logger.get_dir(), "monitor_orig.json"), allow_early_resets=True)
                orig_env.seed(args.seed)

                # add names for the viewers
                if hasattr(env.env.env, 'name'):
                    env.env.env.name = "CAT"
                if hasattr(orig_env.env.env, 'name'):
                    orig_env.env.env.name = "original"
                func_dict['orig_env'] = orig_env

            func_dict['add_noise'] = False
            set_global_seeds(args.seed)
            env.seed(args.seed)
            runner(*func_list, **func_dict)
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_iters, #num_timesteps,
          timesteps_per_batch, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, g_lr, d_lr, batch_size, optim_epochs,
          max_kl=0.01, task_name=None, continue_checkpoint=None,
          semi_dataset=None, freeze_g=False, freeze_d=False):

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        logger.configure(dir=log_dir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    if semi_dataset:
        bc_dataset = semi_dataset
        semi_size = semi_dataset.semi_size
    else:
        bc_dataset = dataset
        semi_size = 0

    pretrained_weight = None
    if continue_checkpoint is not None:
        pretrained_weight = tf.train.latest_checkpoint(continue_checkpoint)
    elif pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from cat_dauggi import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, bc_dataset,
                                                 max_iters=BC_max_iter,
                                                 ckpt_dir=osp.join(checkpoint_dir, "BC"),
                                                 task_name='BC_seed_' + str(MPI.COMM_WORLD.Get_rank()),
                                                 semi_size=semi_size,
                                                 log_dir=log_dir,
                                                 verbose=True)

    if pretrained_weight and freeze_d and freeze_g:
        print("Both G and D instructed to freeze. No training possible... Terminating...")
        return

    if algo == 'trpo':
        from cat_dauggi import trpo_mpi
        # Set up for MPI seed
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_iters=num_iters,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=timesteps_per_batch,
                       max_kl=max_kl, cg_iters=10, cg_damping=0.1,
                       gamma=0.995, lam=0.97,
                       vf_iters=optim_epochs, vf_stepsize=g_lr,
                       d_stepsize=d_lr,
                       task_name=task_name,
                       freeze_g=freeze_g, freeze_d=freeze_d,
                       vf_batchsize=batch_size,
                       semi_dataset=semi_dataset,
                       semi_loss=bool(semi_dataset))        # ADD L2 loss to G reward

    elif algo == 'ppo':
        raise NotImplementedError
    else:
        raise NotImplementedError


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False, render=True, network_prefix='cat_dauggi', semi_dataset=None, orig_env=None,
           find_checkpoint=True, add_noise=False):

    reward_threshold = -1   # if positive, filter out trajectories with reward < reward_threshold
    success_only = False    # filter out trajectories and save only successes

    # Setup network
    # ----------------------------------------
    if network_prefix and network_prefix != 'il' and network_prefix != 'semi':
        print("Invalid network name, terminating...")
        return None

    if semi_dataset:
        from cat_dauggi.semi_utils import semi_ob_space
        ob_space = semi_ob_space(env, semi_dataset.semi_size)
    else:
        ob_space = env.observation_space

    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)

    # Prepare for rollouts
    # ----------------------------------------
    checkpoint_file = tf.train.latest_checkpoint(load_model_path) if find_checkpoint else load_model_path
    is_vaild = True  # checkpoint is valid
    if not U.load_checkpoint_variables(checkpoint_file):
        if network_prefix:
            check_prefix = get_il_prefix() if network_prefix == 'cat_dauggi' else get_semi_prefix()
            if not U.load_checkpoint_variables(checkpoint_file, check_prefix=check_prefix):
                is_vaild = False
        else:
            is_vaild = False

    if not is_vaild:
        # invalid checkpoint
        print("Invalid checkpoint at ", load_model_path)
        return None

    U.initialize()

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    trajectories = {}
    res_dict = {}  # return dict
    orig_res_dict = None
    if hasattr(env.env, 'print_msg'):
        env.env.print_msg = True
    for _ in tqdm(range(number_trajs), ascii=True):
        while True:
            traj, traj_dict, orig_traj_dict = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy, render=render,
                                    semi_dataset=semi_dataset, orig_env=orig_env, add_noise=add_noise)
            # print("\nreward:", traj['ep_rets'])
            if success_only and 'success' in traj and traj['success']:
                break
            if not success_only and (reward_threshold < 0 or traj['ep_rets'] >= reward_threshold):
                break
        obs, acs, ep_len, ep_ret = traj['obs'], traj['acs'], traj['lens'], traj['ep_rets']
        for key in traj:
            if key not in trajectories:
                trajectories[key] = []
            trajectories[key].append(traj[key])
        obs_list.append(obs)
        acs_list.append(acs)

        len_list.append(ep_len)
        ret_list.append(ep_ret)
        for key in traj_dict.keys():
            if key not in res_dict:
                res_dict[key] = 0
            res_dict[key] += float(traj_dict[key])  # num of successes
        if isinstance(orig_traj_dict, dict):
            if not orig_res_dict:
                orig_res_dict = {}
            for key in orig_traj_dict.keys():
                if key not in orig_res_dict:
                    orig_res_dict[key] = 0
                orig_res_dict[key] += float(orig_traj_dict[key])
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')

    for key in trajectories:
        trajectories[key] = np.asarray(trajectories[key])

    if save:
        filename = osp.join(load_model_path, 'policy_traj_' + str(number_trajs) + "_" + env.spec.id)
        np.savez(filename, **trajectories)
        print("saved at", filename)
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    print("number of trajectories:", len(len_list))

    for key in res_dict.keys():
        if res_dict[key] is not None:
            print(key + " policy total = " + str(res_dict[key]), " out of", len(obs_list))
            res_dict[key] = float(res_dict[key] / len(obs_list) * 100)
            print(key + " policy percentage = " + str(res_dict[key]) + "%")
        if orig_res_dict and key in orig_res_dict:
            print(key + " original total = " + str(orig_res_dict[key]) + " out of", len(obs_list))
            orig_res_dict[key] = float(orig_res_dict[key] / len(obs_list) * 100)
            print(key + " original percentage = " + str(orig_res_dict[key]) + "%")

    return avg_len, avg_ret, res_dict


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic, render, semi_dataset=None, orig_env=None, add_noise=False):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []
    qpos = []
    qvel = []
    traj = {}

    res_keys = ['success']
    res_dict = {}
    orig_res_dict = None
    if orig_env is None:
        full_ob, ob, env = reset_env(semi_dataset, env, random_init=False, traj_id=-1, add_noise=add_noise)
    else:
        full_ob, ob, [env, orig_env] = reset_envs(semi_dataset, [env, orig_env], random_init=False, traj_id=-1, add_noise=add_noise)

    if render:
        env.render()
        if orig_env:
            orig_env.render()

    done = False
    while True:
        ac, vpred = pi.act(stochastic, full_ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        if hasattr(env.env.env, 'gs'):
            state = env.env.env.gs()
            for key in state:
                if key not in traj:
                    traj[key] = []
                traj[key].append(state[key])
        else:
            qpos.append(deepcopy(env.env.env.data.qpos))
            qvel.append(deepcopy(env.env.env.data.qvel))

        ob, rew, new, frame_dict = env.step(ac)
        if orig_env is not None and semi_dataset is not None:
            orig_ac = semi_dataset.full_ob_2_acs(full_ob)
            _, _, _, orig_frame_dict = orig_env.step(orig_ac)
            if not orig_res_dict:
                orig_res_dict = {}
            for key in res_keys:
                if key == "success" and hasattr(orig_env.env.env, 'success'):
                    orig_res_dict[key] = orig_env.env.env.success
                elif key in orig_frame_dict.keys():
                    orig_res_dict[key] = orig_frame_dict[key]

            if hasattr(orig_env, 'needs_reset') and orig_env.needs_reset:
                orig_env.needs_reset = False
        if semi_dataset is not None and hasattr(env, 'needs_reset') and env.needs_reset:
            env.needs_reset = False

        for key in res_keys:
            if key == "success" and hasattr(env.env.env, 'success'):
                res_dict[key] = env.env.env.success
            elif key in frame_dict.keys():
                res_dict[key] = frame_dict[key]

        if render:
            env.render()
            if orig_env is not None:
                orig_env.render()

        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if semi_dataset:
            new = False
        if new or t >= horizon or (semi_dataset is not None and done):  # done is for semi network
            break
        else:
            if semi_dataset:
                # is semi network - get the next frame full obs from the trajectory
                full_ob, done = semi_dataset.get_next_frame_full_ob(ob)
            else:
                full_ob = ob
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)

    traj.update({"obs": obs, "rews": rews, "new": news, "acs": acs,
                 "ep_rets": cur_ep_ret, "lens": cur_ep_len})
    if hasattr(env.env.env, 'gs'):
        state = env.env.env.gs()
        for key in state:
            traj[key] = np.asarray(traj[key])
    else:
        traj.update({"qpos": qpos, "qvel": qvel})

    for key in res_keys:
        if key in res_dict.keys():
            traj[key] = res_dict[key]
    return traj, res_dict, orig_res_dict


if __name__ == '__main__':
    args = argsparser()
    main(args)


