__author__ = 'DafniAntotsiou'

'''
This script calculates dataset DTW scor.
It uses the fastdtw python package @https://pypi.org/project/fastdtw/
'''

from cat_dauggi.functions import read_npz
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import argparse
from baselines.common.misc_util import boolean_flag
import numpy as np
import gym
import gym_ext
import adroit_hand_ext
from scipy.stats import zscore
import csv


def argsparser():
    parser = argparse.ArgumentParser("diversity score of trajectories")
    parser.add_argument('--traj_1', help='1st trajectory',
                        # default='data/door/augmented_trajectories_1_0.0.npz')
                        default="gail/door_ext_300-v0/trpo_gail.semi.with_pretrained.BC1e+04.8e-04.1e-04.256.transition_limitation_-1.door_ext_300.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001.seed_1/hour040/policy_traj_1_door_ext_300-v0.npz")

    parser.add_argument('--traj_2', help='2nd trajectory',
                        default='data/door/augmented_trajectories_1_0.2.npz')

    parser.add_argument('--trajectories', help='trajectories dataset',
                        # default=None)
                        default="gail/hammer_ext-v0/dynamic_experts/rew_sparse/ms_trpo_gail.continue.8e-04.1e-04.256.transition_limitation_-1.hammer_ext.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001.seed_1/hour106/policy_traj_100_hammer_ext-v0.npz")
    parser.add_argument('--traj_limitation', help='number of trajectories', default=-1, type=int)
    boolean_flag(parser, 'render', default=True, help='render the trajectories')
    parser.add_argument('--env_id', help='environment ID', default='HalfCheetah_ext-v2')
    return parser.parse_args()


def dataset_diversity_score(data, fulldtw=False):
    score = None
    frame_score = None
    if 'obs' in data:
        score = 0
        frame_score = 0
        trajectories = data['obs']
        print(len(trajectories[0]))
        for i in range(len(trajectories)-1):
            traj_i = np.nan_to_num(zscore(trajectories[i]))   # z score on axis=0
            for j in range(i+1, len(trajectories)):
                # z normalise the trajectory
                traj_j = np.nan_to_num(zscore(trajectories[j]))
                # print(len(traj_i),len(traj_j))
                radius = 1 if not fulldtw else int(min(max(len(trajectories[i]), len(trajectories[j]))/4, 10))
                curr_score = fastdtw(traj_i, traj_j, dist=euclidean, radius=radius)[0]
                score += curr_score
                frame_score += curr_score / (len(traj_i) + len(traj_j))
        if len(trajectories):
            div = (len(trajectories) - 1) * len(trajectories) / 2
            score /= div
            frame_score /= div
    return score, frame_score


def mean_trajectory(traj_1, traj_2, w_1=1, w_2=1, threshold=None, radius=1):
    """ Get the mean DTW trajectory between two trajectories

    traj_1 : the first trajectory
    traJ_2 : the steps of the second trajectory
    w_1 : weight of first trajectory for running mean
    w_2 : weight of second trajectory for running mean
    threshold : the score threshold above which the combination takes place.
    Return : the steps of the mean trajectory and the dtw score between the two trajectories
    """

    score, indeces = fastdtw(traj_1, traj_2, dist=euclidean, radius=radius)
    mean_traj = None
    if threshold is None or score > threshold:
        mean_traj = []
        for pair in indeces:
            pair_mean = (float(w_1) * traj_1[pair[0]] + float(w_2) * traj_2[pair[1]]) / float(w_1 + w_2)
            mean_traj.append(pair_mean)
        mean_traj = np.asanyarray(mean_traj)
    return mean_traj, score


def test_combination(traj_1, traj_2):
    mean_traj, score = mean_trajectory(traj_1, traj_2, 1, 1)
    print("original dtw score", score)
    m = np.mean(traj_1 - traj_2)
    print("original mean score", m)
    i = 1
    while True:
        i += 1
        mean_traj, score = mean_trajectory(mean_traj, traj_2, i, 1)
        print("new dtw score", score)
        m = np.mean((mean_traj * i - traj_2)/(i+1))
        print("new mean score", m)


def test_combinations(trajectories):
    assert len(trajectories) > 0
    mean_traj = trajectories[0]
    for i in range(1, len(trajectories)):
        mean_traj, score = mean_trajectory(mean_traj, trajectories[i], i, 1)
        print("dtw score for i =", i, score)
        _, score2 = mean_trajectory(trajectories[0], trajectories[i], 1, 1)
        print("dtw with 1st trajectory for i =", i, score2)


def main(args):
    if args.trajectories is not None:
        data = read_npz(args.trajectories)
    elif args.traj_1 is not None and args.traj_2 is not None:
        data1 = read_npz(args.traj_1)
        data2 = read_npz(args.traj_2)
        data = {}
        for label in data1:
            if label in data2:
                data[label] = np.array([data1[label].squeeze(), data2[label].squeeze()])

    if args.traj_limitation > 0:
        for key in data:
            if len(data[key]) > args.traj_limitation:
                data[key] = data[key][:args.traj_limitation]

    if 'ep_rets' in data:
        print("original data mean ret:", np.mean(data["ep_rets"]), "min:", np.min(data["ep_rets"]), "max:", np.max(data["ep_rets"]))

    diversity_score, frame_div_score = dataset_diversity_score(data, fulldtw=True)
    if diversity_score is not None:
        print("original dataset mean diversity dtw: ", diversity_score)
    if frame_div_score is not None:
        print("original dataset mean frame diversity dtw: ", frame_div_score)
    if args.trajectories is not None:
        res_file = args.trajectories.rstrip(".npz") + ".csv"
        with open(res_file, mode='w', newline='') as step_file:
            writer = csv.writer(step_file)
            name_list = [args.trajectories, str(diversity_score), str(frame_div_score)]
            writer.writerow(name_list)

    if args.render and len(data['obs']) == 2:   # render traj side-by-side for visual comparison if there are only 2
        env = []
        max_len = max([len(i) for i in data['obs']])
        for traj in range(len(data['obs'])):
            env.append(gym.make(args.env_id))

        while True:
            for traj in range(len(data['acs'])):
                env[traj].reset()
                if hasattr(env[traj].env, 'ss'):
                    # get dictionary of initial frame
                    state_dict = {key: data[key][traj][0] for key in data if key != 'ep_rets' and key != "lens"}
                    env[traj].env.ss(state_dict)
                elif 'qpos' in data.keys() and 'qvel' in data.keys():
                    qpos = data['qpos'][traj][0]
                    qvel = data['qvel'][traj][0]
                    env[traj].env.set_state(qpos, qvel)
                env[traj].render()
            for i in range(0, max_len):
                for traj in range(len(env)):
                    if i < len(data['acs'][traj]):
                        acs = (data['acs'][traj][i])
                        observation, reward, done, info = env[traj].step(acs)
                        env[traj].render()


if __name__ == "__main__":
    args = argsparser()
    main(args)
