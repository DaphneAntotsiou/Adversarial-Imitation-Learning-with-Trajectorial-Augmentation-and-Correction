__author__ = 'DafniAntotsiou'

'''This is heavily based on behavior_clone of @openai/baselines.gail'''

'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''

import argparse
import tempfile
import os.path as osp
import gym
import logging
from tqdm import tqdm

import tensorflow as tf

from cat_dauggi.mlp_policy_ext import MlpPolicy as mlp_policy
from baselines import bench
from cat_dauggi import logger
from baselines.common import set_global_seeds
from baselines.common.misc_util import boolean_flag
from baselines.common.mpi_adam import MpiAdam
from cat_dauggi.run_gail import runner
from cat_dauggi.mujoco_dset import Mujoco_Dset
import cat_dauggi.tf_util as U
import os
from cat_dauggi.mlp_policy_ext import MlpPolicy_Custom as mlp_custom
import numpy as np
from gym import spaces
from cat_dauggi.semi_utils import semi_ob_space
from mpi4py import MPI
from collections import deque
from cat_dauggi.statistics import stats


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Behavior Cloning")
    parser.add_argument('--env_id', help='environment ID', default='HalfCheetah_ext-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--expert_path', type=str, default='data/HalfCheetah-v2/trajectories.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default=None)

    parser.add_argument('--load_model_path', help='dir to load the trajectories for evaluation', type=str,
                        default="bc/107000_rl_4_gail_10/BC.HalfCheetah.traj_limitation_-1.seed_1")
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=64)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=True, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=100000)
    parser.add_argument('--policy', type=str, choices=['default'], help='policy architecture', default='default')
    parser.add_argument('--continue_checkpoint', help='continue training the latest checkpoint', type=str,
                        default=None)
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='evaluate')
    boolean_flag(parser, 'eval_hours', default=False, help='evaluate all the steps of a train run')
    boolean_flag(parser, 'render', default=False, help='render trajectories for qualitative evaluation')
    boolean_flag(parser, 'semi', default=False, help='Dummy parameter - does nothing')

    return parser.parse_args()


def learn(env, policy_func, dataset, optim_batch_size=128, max_iters=1e4,
          adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False, pretrained_weight=None, prefix='', semi_size=0):

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        logger.configure(dir=log_dir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    ob_space = semi_ob_space(env, semi_size=semi_size)

    val_per_iter = min(int(max_iters/10), 1000)

    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, prefix=prefix)  # Construct network for new policy
    # placeholder
    ob = U.get_placeholder_cached(name=prefix + "ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name=prefix + "stochastic")
    loss = tf.reduce_mean(tf.square(ac-pi.ac))
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()

    if rank == 0 and log_dir is not None:
        train_loss_buffer = deque(maxlen=40)
        eval_loss_buffer = deque(maxlen=40)
        train_stats = stats(["Train_Loss"], scope="Train_Loss")
        val_stats = stats(["Evaluation_Loss"], scope="Validation_Loss")
        writer = U.file_writer(log_dir)

    # load previous checkpoint if specified
    if pretrained_weight is not None:
        exclude = []
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=exclude)

        if tf.train.checkpoint_exists(pretrained_weight):
            init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
                pretrained_weight,
                variables_to_restore, ignore_missing_vars=True)
        else:
            print('No valid checkpoint to restore')

        U.get_session().run(tf.global_variables_initializer())
        if tf.train.checkpoint_exists(pretrained_weight):
            U.get_session().run(init_assign_op, init_feed_dict)

    prev_val_loss = 1e10
    logger.log("Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        train_loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)
        if rank == 0 and log_dir is not None:
            train_stats.add_all_summary(writer, [train_loss], iter_so_far)
            if not iter_so_far % val_per_iter:
                logger.record_tabular("IterationsSoFar", iter_so_far)
                logger.record_tabular("TrainLoss", train_loss)

        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))
            loss_diff = prev_val_loss - val_loss
            prev_val_loss = val_loss
            logger.log("Validation loss reduced by ", str(loss_diff))
            if rank == 0 and log_dir is not None:
                val_stats.add_all_summary(writer, [val_loss], iter_so_far)
                logger.record_tabular("ValidationLoss", val_loss)

            # if abs(loss_diff) < 0.0005:
            #     print("no change in evaluation loss, terminating")
            #     break

        if rank == 0 and log_dir is not None and not iter_so_far % val_per_iter:
            logger.dump_tabular()

        if ckpt_dir is not None and rank == 0 and iter_so_far % 5000 == 0:
            os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
            savedir_fname = osp.join(ckpt_dir, task_name + "_" + str(iter_so_far).zfill(5))
            U.save_no_prefix(savedir_fname, pi.get_variables(), prefix=prefix, write_meta_graph=False)

    if ckpt_dir is None or rank != 0:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        savedir_fname = osp.join(ckpt_dir, task_name + "_iter_" + str(max_iters).zfill(5))
        U.save_no_prefix(savedir_fname, pi.get_variables(), prefix=prefix, write_meta_graph=False)

        savedir_fname = osp.join(ckpt_dir, task_name + "_final")

    U.save_no_prefix(savedir_fname, pi.get_variables(), prefix=prefix, write_meta_graph=False)
    return savedir_fname


def get_task_name(args):
    task_name = 'BC'
    task_name += '.{}'.format(args.env_id.split("-")[0])
    task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    task_name += ".seed_{}".format(args.seed)
    return task_name


def main(args):
    U.make_session().__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    def policy_fn(name, ob_space, ac_space, reuse=False, prefix=''):
        if args.policy == 'default':
            return mlp_policy(name=name, ob_space=ob_space, ac_space=ac_space,
                              hid_size=args.policy_hidden_size, num_hid_layers=2, prefix=prefix)
        else:
            return mlp_custom(name=name, ob_space=ob_space, ac_space=ac_space)

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name) if args.log_dir else osp.join(args.checkpoint_dir, 'log')
    if args.task == "train":
        dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)

        pretrained_weight = None
        if args.continue_checkpoint is not None:
            pretrained_weight = tf.train.latest_checkpoint(args.continue_checkpoint)

        savedir_fname = learn(env,
                              policy_fn,
                              dataset,
                              max_iters=args.BC_max_iter,
                              ckpt_dir=args.checkpoint_dir,
                              log_dir=args.log_dir,
                              task_name=task_name,
                              verbose=True,
                              pretrained_weight=pretrained_weight)
    elif args.task == 'evaluate':
        if args.env_id.find("hand") != -1:
            env.env.env.epsilon = 0  # always start from sigma random init
        func_list = [env, policy_fn]
        func_dict = {'load_model_path': args.load_model_path,
                     'timesteps_per_batch': 1024,
                     'number_trajs': 100,
                     'stochastic_policy': args.stochastic_policy,
                     'save': args.save_sample,
                     'render': args.render,
                     'find_checkpoint': True}
        if args.eval_hours:
            from cat_dauggi.eval_utils import evaluate_runs
            func_dict["find_checkpoint"] = False
            # from cat_dauggi.retargeting_utils import evaluate_steps
            # evaluate_steps(runner, func_list=func_list, func_dict=func_dict, seed=args.seed, append_step=-1)
            evaluate_runs(runner, func_list=func_list, func_dict=func_dict, seed=args.seed, keyword='',
                          dir_separation=False)
        else:
            set_global_seeds(args.seed)
            env.seed(args.seed)
            runner(*func_list, **func_dict)


if __name__ == '__main__':
    args = argsparser()
    main(args)
