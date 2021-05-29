__author__ = 'DafniAntotsiou'
'''
convert mj_envs datasets to be compatible with openai format
'''

from cat_dauggi.functions import read_npz
import argparse
import os
import gym
from copy import deepcopy
import numpy as np
import adroit_hand_ext
from baselines.common.misc_util import boolean_flag


def argsparser():
    parser = argparse.ArgumentParser("replay trajectories")
    parser.add_argument('--env_id', help='environment ID', default='hammer_ext_500-v0')
    parser.add_argument('--expert_path', type=str, help='path of the current trajectories',
                        default="data/hammer-v0_demos.pickle")
    parser.add_argument('--output', help='output filename of converted trajectories', default='trajectories.npz')
    boolean_flag(parser, 'render', default=False, help='render the trajectories')
    return parser.parse_args()


def convert_dataset(env_id, fname):
    """
    Convert the dataset in a compatible form according to the openai datasets
    :param env_id:
    environment the dataset is run onto
    :param fname:
    path to the dataset file
    :return:
    the dataset dictionary with the converted dataset
    """

    data = read_npz(fname)
    if not data:
        # is emtpy
        print("dataset is empty")
        return data
    if isinstance(data, dict) and 'obs' in data and 'acs' in data:
        # data is in correct format
        return data
    elif isinstance(data, list) and isinstance(data[0], dict) and 'observations' in data[0] and 'actions' in data[0] \
            and 'rewards' in data[0]:
        res = {'obs': [], 'acs': [], 'qpos': [], 'qvel': [], 'ep_rets': []}
        env = gym.make(env_id)
        for traj in data:
            env.reset()
            if 'init_state_dict' in traj:
                env.env.ss(traj['init_state_dict'])
                for key in traj['init_state_dict']:
                    if key not in res:
                        res[key] = []        # add state key in main dictionary
                if args.render:
                    env.render()

            traj_res = {v: [] for v in res}  # dictionary for this trajectory
            traj_res['ep_rets'] = 0
            observation = env.env._get_obs()
            for frame in range(0, len(traj['observations'])):
                acs = deepcopy(traj['actions'][frame])

                traj_res['obs'].append(observation)
                traj_res['acs'].append(acs)

                state = env.env.gs()
                for key in state:
                    if key in traj_res:
                        traj_res[key].append(np.array(state[key]))

                # if not np.array_equal(observation, traj['observations'][frame]):
                if (np.absolute(observation-traj['observations'][frame]).max() > 0.1):
                    print("replay observations do not match...")

                observation, reward, done, info = env.step(acs)

                if args.render:
                    env.render()

                if reward != traj['rewards'][frame]:
                    print("replay rewards do not match...")

                traj_res['ep_rets'] += reward

                # if done or frame >= len(traj['observations']) - 1:
                if frame >= len(traj['observations']) - 1:
                    for key in res:
                        if key in traj_res:
                            res[key].append(np.array(traj_res[key]))
        env.close()
        return res
    else:
        print("Unexpected type of data")
        return data


def main(args):
    data = convert_dataset(args.env_id, args.expert_path)
    if data:
        np.savez(args.output, **data)
        print("successful convertion of dataset at " + args.output)


if __name__ == "__main__":
    args = argsparser()
    main(args)
