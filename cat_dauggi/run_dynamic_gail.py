__author__ = 'DafniAntotsiou'

'''
This script is heavily based on @openai/baselines.gail.run_mujoco
'''

'''
DAugGI network train + evaluate 
'''

import argparse
import os.path as osp
import logging
from mpi4py import MPI

import numpy as np
import gym

from cat_dauggi.mlp_policy_ext import MlpPolicy as mlp_policy
from baselines.common import set_global_seeds
from baselines.common.misc_util import boolean_flag
from baselines import bench
from cat_dauggi import logger
import tensorflow as tf
from cat_dauggi import tf_util as U
from cat_dauggi.semi_utils import get_il_prefix, get_semi_prefix
from cat_dauggi.eval_utils import evaluate_runs
from cat_dauggi.semidset import SemiDset
from collections import OrderedDict
try:
    import gym_ext
    import adroit_hand_ext
except ImportError:
    pass


def none_or_str(value):
    if value == 'None':
        return None
    return value


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of DAugGI")
    parser.add_argument('--env_id', help='environment ID', default='HalfCheetah_ext-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)

    parser.add_argument('--semi_path', type=str, default='data/HalfCheetah/augmented_trajectories.npz', help='the directory to the semi dataset')

    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='dauggi/')
    parser.add_argument('--log_dir', help='the directory to save log file', default=None)
    # evaluation
    parser.add_argument('--load_model_path', help='dir to load the trajectories for evaluation', type=str,
                        default=None)
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='evaluate')
    boolean_flag(parser, 'eval_runs', default=False, help='evaluate all the hours of a train run')

    # network to evaluate
    parser.add_argument('--network', type=str, choices=['il', 'semi'], default='il')
    parser.add_argument('--semi_label', type=str, choices=['acs', 'hpe'], default='acs')
    # for evaluation n
    boolean_flag(parser, 'stochastic_policy', default=True, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=True, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)

    boolean_flag(parser, 'freeze_g', default=False, help='freeze generator and train discriminator only')
    boolean_flag(parser, 'freeze_d', default=False, help='freeze discriminator and train generator only')

    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=64)
    parser.add_argument('--adversary_hidden_size', type=int, default=64)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    boolean_flag(parser, 'hybrid', default=False, help='Use hybrid RL + IL reward')     # hybrid reward

    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=1)
    # parser.add_argument('--num_timesteps', help='max number of timesteps', type=int, default=5e50)
    parser.add_argument('--num_iters', help='max number of timesteps', type=int, default=5e50)
    parser.add_argument('--timesteps_per_batch', help='total generator timesteps per batch', type=int, default=int(2 ** 14))  # total number regardless of threads
    parser.add_argument('--continue_checkpoint', help='directory of latest checkpoint to continue', type=none_or_str,
                        default=None)
    parser.add_argument('--continue_il', help='directory of latest DAugGI checkpoint to continue', type=none_or_str,
                        default=None)
    parser.add_argument('--expert_model', help='directory of expert CAT checkpoint', type=str)

    # binary filter for expert generator
    boolean_flag(parser, 'filter_expert', default=True, help='filter the dynamic experts using success binary filter or RL reward')
    boolean_flag(parser, 'sparse_reward', default=True, help='filter using sparse reward')
    parser.add_argument('--reward_thresh', type=float, help='filter expert by RL reward if sparse_reward not set', default=100)

    parser.add_argument('--policy', type=str, choices=['custom', 'default'], help='policy architecture', default='default')
    boolean_flag(parser, 'init_expert_traj', default=True, help='initialise simulation from expert trajectories')
    boolean_flag(parser, 'render', default=False, help='render trajectories for qualitative evaluation')
    parser.add_argument('--g_lr', type=float, help='generator learning rate', default=1e-3)
    parser.add_argument('--d_lr', type=float, help='discriminator learning rate', default=3e-4)
    parser.add_argument('--batch_size', type=float, help='batch size', default=128)
    parser.add_argument('--optim_epochs', type=float, default=5)
    return parser.parse_args()


def get_task_name(args):
    task_name =args.algo + "_dauggi."

    if args.continue_checkpoint or args.continue_il:
        task_name += "continue."

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
    task_name += args.env_id.split("-")[0]
    task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
        ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    task_name += ".seed_" + str(args.seed)
    return task_name


def main(args):
    sess = U.make_session().__enter__()
    set_global_seeds(args.seed)
    semi_env = args.env_id
    # door and hammer expert trajectories are longer than default IL envs, so load those for cat network
    if semi_env.find("door") != -1:
        semi_env = "door_ext_300-v0"
    elif semi_env.find("hammer") != -1:
        semi_env = "hammer_ext_500-v0"
    env = OrderedDict()
    if args.network == 'semi' or args.task == 'train':
        env[get_semi_prefix()] = gym.make(semi_env)
    if args.network == 'il' or args.task == 'train':
        env[get_il_prefix()] = gym.make(args.env_id)

    def policy_fn(name, ob_space, ac_space, reuse=False, prefix=''):
        if args.policy == 'default':
            return mlp_policy(name=name, ob_space=ob_space, ac_space=ac_space,
                              hid_size=args.policy_hidden_size, num_hid_layers=2, prefix=prefix)
        else:
            NotImplementedError

    for label in env:
        env[label] = bench.Monitor(env[label], logger.get_dir() and
                                   osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
        env[label].seed(args.seed)

    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name) if args.log_dir else osp.join(args.checkpoint_dir, 'log')

    if args.task == 'train':
        semi_dataset = SemiDset(expert_path=args.semi_path, traj_limitation=-1, seed=args.seed,
                                semi_label=args.semi_label)  # it can be hpe or acs
        from cat_dauggi.adversary_ext import TransitionClassifier_Ext as TransitionClassifier

        reward_giver = TransitionClassifier(env[get_il_prefix()], args.adversary_hidden_size, entcoeff=args.adversary_entcoeff) #, labels=list(env.keys()))
        train(env,
              args.seed,
              policy_fn,
              reward_giver,
              semi_dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_iters,
              args.timesteps_per_batch,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.g_lr,
              args.d_lr,
              args.batch_size,
              args.optim_epochs,
              args.max_kl,
              task_name,
              args.continue_checkpoint,
              args.continue_il,
              args.expert_model,
              freeze_d=args.freeze_d,
              freeze_g=args.freeze_g,
              expert_threshold=args.reward_thresh if args.filter_expert and not args.sparse_reward else None,
              sparse_reward=args.sparse_reward if args.filter_expert else False
              )
    elif args.task == 'evaluate':
        from cat_dauggi.run_gail import runner
        semi_dataset = SemiDset(expert_path=args.semi_path, traj_limitation=args.traj_limitation,
                                randomize=True, seed=args.seed, semi_label=args.semi_label) \
            if args.network == 'semi' else None

        eval_env = list(env.values())[0]

        func_list = [eval_env, policy_fn]
        func_dict = {'load_model_path': args.load_model_path,
                     'timesteps_per_batch': eval_env.spec.max_episode_steps,
                     'number_trajs': 100,
                     'stochastic_policy': args.stochastic_policy,
                     'save': args.save_sample,
                     'render': args.render,
                     'semi_dataset': semi_dataset,
                     'network_prefix': args.network}
        if args.eval_runs:
            func_dict["save"] = False

            keyword = 'hour'
            step = 1
            if args.env_id.find("Pendulum") != -1:
                keyword = "iter_"
                step = 10
            evaluate_runs(runner, func_list=func_list, func_dict=func_dict, seed=args.seed, keyword=keyword, step=step,
                          stochastic=args.stochastic_policy, continue_hour=True, dir_separation=True)
        else:
            if semi_dataset is not None:
                # add second environment for the original noisy trajectories
                orig_env = gym.make(args.env_id)
                orig_env = bench.Monitor(orig_env, logger.get_dir() and
                                         osp.join(logger.get_dir(), "monitor_orig.json"), allow_early_resets=True)
                orig_env.seed(args.seed)
                func_dict['orig_env'] = orig_env
            set_global_seeds(args.seed)
            for label in env:
                env[label].seed(args.seed)
            runner(*func_list, **func_dict)

    else:
        raise NotImplementedError
    for label in env:
        env[label].close()


def train(env, seed, policy_fn, reward_giver,  # dataset,
          semi_dataset, algo,
          g_step, d_step, policy_entcoeff, num_iters,  # num_timesteps,
          timesteps_per_batch, save_per_iter,
          checkpoint_dir, log_dir, g_lr, d_lr, batch_size, optim_epochs,
          max_kl=0.01, task_name=None, continue_checkpoint=None,
          continue_il=None, expert_model=None, freeze_d=False, freeze_g=False, expert_threshold=None,
          expert_label=get_semi_prefix(), sparse_reward=True):
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        logger.configure(dir=log_dir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    pretrained_weight = None
    pretrained_il = None
    pretrained_semi = None
    if continue_checkpoint is not None:
        pretrained_weight = tf.train.latest_checkpoint(continue_checkpoint)
    elif continue_il is not None or expert_model is not None:
        # get pretrained weights for the IL - DAugGI network
        if continue_il is not None:
            pretrained_il = tf.train.latest_checkpoint(continue_il)
        # get pretrained weights for the semi - CAT network
        if expert_model is not None:
            pretrained_semi = tf.train.latest_checkpoint(expert_model)

    if pretrained_weight and freeze_d and freeze_g:
        print("Both G and D instructed to freeze. No training possible... Terminating...")
        return

    if algo == 'trpo':
        from cat_dauggi import trpo_dynamic_expert as trpo_mpi
        # Set up for MPI seed
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)

        for label in env:
            env[label].seed(workerseed)

        trpo_mpi.learn(env, policy_fn, reward_giver,
                       semi_dataset, rank,
                       pretrained_weight=pretrained_weight,
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
                       pretrained_il=pretrained_il,
                       pretrained_semi=pretrained_semi,
                       semi_loss=bool(semi_dataset),  # ADD L2 loss to G reward
                       expert_reward_threshold=expert_threshold,  # filter experts on environmental reward
                       expert_label=expert_label,  # define which network is the teacher
                       sparse_reward=sparse_reward
                       )

    elif algo == 'ppo':
        NotImplementedError
    else:
        raise NotImplementedError
