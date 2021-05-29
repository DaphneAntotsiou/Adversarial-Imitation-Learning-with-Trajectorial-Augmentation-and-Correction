__author__ = 'DafniAntotsiou'
'''
augment trajectories by injecting them with artificial noise in the action space
'''

import gym_ext
import adroit_hand_ext

import gym
import argparse
import numpy as np
import os
from copy import deepcopy
from baselines.common.misc_util import boolean_flag
from gym.utils import seeding

from cat_dauggi.functions import read_npz


def argsparser():
    parser = argparse.ArgumentParser("augment trajectories")
    parser.add_argument('--env_id', help='environment ID', default='InvertedPendulum_ext-v2')
    parser.add_argument('--expert_path', type=str, help='path of the current trajectories',
                        default='data/InvertedPendulum/policy_traj_100_InvertedPendulum-v2.npz')
    parser.add_argument('--sigma', type=str, help='sigma of noise distribution', default=0.9)
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--traj_limitation', type=int, default=7, help="trajectories in expert dataset")
    parser.add_argument('--trajectories', help='number of augmented trajectories for each existing trajectory',
                        type=int, default=71)
    boolean_flag(parser, 'mocap_only', default=False, help='use only mocap for actions')
    boolean_flag(parser, 'render', default=False, help='render the trajectories')
    parser.add_argument('--state_norm_path', type=str, help='normalise observations based on distribution',
                        default=None)
    return parser.parse_args()


def main(args):
    data = read_npz(args.expert_path)

    if args.traj_limitation > 0:
        for key in data:
            if len(data[key]) > args.traj_limitation:
                data[key] = data[key][:args.traj_limitation]

    env = gym.make(args.env_id)
    orig_env = gym.make(args.env_id)

    np_random, seed = seeding.np_random(args.seed)

    res = {}
    for key in data.keys():
        res[key] = []
    new_keys = ['qpos', 'qvel', 'ep_rets', 'obs', 'acs', 'lens']      # keys that will be updated and not copied from data

    for key in new_keys:
        res[key] = []

    if 'obs' in data.keys() and 'acs' in data.keys():
        for traj in range(len(data['acs'])):    # iterate trajectories
            print("trajectory " + str(traj))
            for augment in range(args.trajectories):
                print("augmentation ", augment + 1)

                env.reset()
                orig_env.reset()

                # initial state of the trajectory
                if hasattr(env.env, 'ss'):
                    # get dictionary of initial frame
                    state_dict = {key: data[key][traj][0] for key in data if key != 'ep_rets' and key != 'lens'}
                    env.env.ss(state_dict)
                else:
                    # only for gym_ext...
                    if 'qpos' in data.keys() and 'qvel' in data.keys():
                        qpos = data['qpos'][traj][0]
                        qvel = data['qvel'][traj][0]
                        env.env.set_state(qpos, qvel)

                if hasattr(orig_env.env, 'ss'):
                    # get dictionary of initial frame
                    state_dict = {key: data[key][traj][0] for key in data if key != 'ep_rets' and key != 'lens'}
                    orig_env.env.ss(state_dict)
                else:
                    # only for gym_ext...
                    if 'qpos' in data.keys() and 'qvel' in data.keys():
                        qpos = data['qpos'][traj][0]
                        qvel = data['qvel'][traj][0]
                        orig_env.env.set_state(qpos, qvel)

                if args.render:
                    env.render()
                    orig_env.render()

                traj_res = {key: [] for key in new_keys}    # new keys for this trajectory
                traj_res['ep_rets'] = 0

                step = 1
                for i in range(0, len(data['acs'][traj]), step):  # same fps as the original trajectories

                    # traj_res['obs'].append(env.env.state_vector())
                    traj_res['obs'].append(env.env._get_obs())

                    acs = deepcopy(data['acs'][traj][i])
                    _ = orig_env.step(acs)

                    # add the noise!!!
                    acs += np_random.uniform(low=-args.sigma, high=args.sigma, size=len(acs))

                    traj_res['qpos'].append(deepcopy(env.env.sim.data.qpos[:]))
                    traj_res['qvel'].append(deepcopy(env.env.sim.data.qvel[:]))

                    acs = acs[:len(env.env.action_space.low)]   # clip the actions to the environment's action space

                    traj_res['acs'].append(acs)

                    observation, reward, done, info = env.step(acs)

                    if args.render:
                        env.render()
                        orig_env.render()

                    # print(reward)

                    traj_res['ep_rets'] += reward
                    done = False
                    if done or i + step >= len(data['acs'][traj]):
                        print("terminated at step ", i + 1, " with trajectory length ", len(data['acs'][traj]))
                        traj_res['lens'] = i + 1
                        # save this trajectory up to this point
                        for key in data:
                            if key in new_keys:  # key is updated - don't copy from data
                                continue
                            if key in res.keys():
                                res[key].append(data[key][traj][:i+1:step])
                            else:
                                res[key] = [data[key][traj][:i+1:step]]

                        # add the new keys
                        for key in new_keys:
                            if key == 'ep_rets' or key == 'lens':
                                res[key].append(traj_res[key])
                            else:
                                res[key].append(np.asarray(traj_res[key]))
                        break

    print("total augmented trajectories produced: ", len(res['obs']))
    file_name = os.path.join(os.path.dirname(args.expert_path),
                             "augmented_trajectories_" + str(args.traj_limitation) + "_" +
                             str(len(res["obs"])) + "_" + str(args.sigma) + ".npz")
    np.savez(file_name, **res)
    env.close()
    orig_env.close()
    print("saved in ", file_name)
    if len(res['ep_rets']):
        avg_ret = sum(res['ep_rets']) / len(res['ep_rets'])
        print("Average return:", avg_ret)
    if len(data["ep_rets"]):
        avg_ret = sum(data['ep_rets']) / len(data['ep_rets'])
        print("Original Average return:", avg_ret)


if __name__ == "__main__":
    args = argsparser()
    main(args)
