__author__ = 'DafniAntotsiou'

import numpy as np
from gym import spaces


# universal prefix for the different networks
def get_il_prefix():
    return 'il_'


def get_semi_prefix():
    return 'semi_'


# add auxiliary actions to env observation space for semi network
def semi_ob_space(env, semi_size):
    if semi_size > 0:
        semi_obs_dim = env.observation_space.shape[0] + semi_size
        semi_high = np.inf * np.ones(semi_obs_dim)
        semi_low = -semi_high
        return spaces.Box(semi_low, semi_high, dtype=np.float64)
    else:
        return env.observation_space


def reset_envs(semi_dataset, envs, traj_id=-1, random_init=True, add_noise=False):
    """
    resets the environment to a frame in the semi-supervised dataset
    :param semi_dataset: the dataset
    :param env: List of environments to be reset.
    :param traj_id: the id number of the trajectory to initialise the environment to. Is random if < 0.
    :param random_init: Initialise at the beginning of the trajectory if False, random trajectory frame if True.
    :param add_noise: add noise to the dataset trajectories during reset
    :return: the (full_ob of the semi-supervised network, the environment ob, the environment) tuple
    """
    # reset the first env in list and then copy that to the rest of the envs
    full_ob, ob, set_env = reset_env(semi_dataset, envs[0], traj_id=traj_id, random_init=random_init, add_noise=add_noise)
    qpos = set_env.env.env.sim.get_state().qpos.copy()
    qvel = set_env.env.env.sim.get_state().qvel.copy()
    if hasattr(set_env.env.env, "gs"):
        semi_dict = set_env.env.env.gs()
    for env in envs:
        if env == set_env:
            continue
        env.reset()
        if hasattr(env.env.env, "ss"):
            if add_noise:
                env.env.env.ss(semi_dict, add_noise=add_noise)  # written like this for debug. TODO: refactor
            else:
                env.env.env.ss(semi_dict)
        elif hasattr(env.env.env, "reset_model_pos"):
            env.env.env.reset_model_pos(qpos=qpos, qvel=qvel)

        elif hasattr(env.env.env, "set_state"):
            env.env.env.set_state(qpos=qpos, qvel=qvel)
        else:
            print("Incompatible environment for semi supervision...")
            exit(1)

    return full_ob, ob, envs


def reset_env(semi_dataset, env, traj_id=-1, random_init=True, add_noise=False):
    """
    resets the environment to a frame in the semi-supervised dataset
    :param semi_dataset: the dataset
    :param env: the environment to be reset
    :param traj_id: the id number of the trajectory to initialise the environment to. Is random if < 0.
    :param random_init: Initialise at the beginning of the trajectory if False, random trajectory frame if True.
    :param add_noise: add noise to the semi_dataset trajectories during reset
    :return: the (full_ob of the semi-supervised network, the environment ob, the environment) tuple
    """
    ob = env.reset()
    if semi_dataset:
        # is the retargeting network - reset the env with semi-labels
        semi_dict = semi_dataset.init_traj_labels(traj_id=traj_id, random_init=random_init)  # random initialisation
        if not semi_dict:
            print("No available expert semi-labels fam!")
            exit(1)

        # reset the environment with the observations from the dataset
        if hasattr(env.env.env, "ss"):
            if add_noise:
                env.env.env.ss(semi_dict, add_noise=add_noise)  # written like this for debug. TODO: refactor
            else:
                env.env.env.ss(semi_dict)
        elif hasattr(env.env.env, "reset_model_pos"):
            env.env.env.reset_model_pos(qpos=semi_dict['qpos'], qvel=semi_dict['qvel'])
        elif hasattr(env.env.env, "set_state"):
            env.env.env.set_state(qpos=semi_dict['qpos'], qvel=semi_dict['qvel'])
        else:
            print("Incompatible environment for retargeting...")
            exit(1)
        full_ob = semi_dict['full_ob']
        ob = semi_dict['ob']
    else:
        full_ob = ob

    return full_ob, ob, env


def semi_loss_func(ac, full_ob, semi_dataset, is_relative_actions=False):
    """
    get the L2 loss between generated actions and semi supervised actions
    :param ac: the semi supervised actions
    :param full_ob: the full observations of the semi supervised dataset
    :param semi_dataset: the semi supervised dataset
    :return: the L2 loss if semi_dataset exists, 0 otherwise
    """
    diff = ac - semi_dataset.full_ob_2_acs(full_ob) if not is_relative_actions else ac
    return (diff ** 2).mean() if semi_dataset is not None else 0


def relative_2_absolute_action(ac, full_ob, semi_dataset, ac_space):
    """
    get absolute action by adding the relative action to the original from the semi dataset, given environmental action bounds
    :param ac: the relative actions from the policy
    :param full_ob: the full set of observations from semi_dataset that produced ac
    :param semi_dataset: the semi dataset that produced the full_ob
    :param ac_space: the action space of the environment, to set the action bounds
    :return: the absolute value of the actions to apply to the environment
    """
    orig_ac = semi_dataset.full_ob_2_acs(full_ob)
    sigma_ratio = 0.4   # ratio of sigma ac can move orig_ac in both directions
    sigma = (ac_space.high - ac_space.low) * sigma_ratio
    ac = np.clip(ac, -sigma, sigma)
    return np.clip(ac + orig_ac, ac_space.low, ac_space.high)

