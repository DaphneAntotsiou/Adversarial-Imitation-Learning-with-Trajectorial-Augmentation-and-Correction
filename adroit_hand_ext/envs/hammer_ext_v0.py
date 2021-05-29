__author__ = 'DafniAntotsiou'

"""
This script is an extension of HammerEnvV0 from mj_envs
"""
from gym import error
try:
    from mujoco_py import MjSim
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

from mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0
import numpy as np


class HammerEnvExtV0(HammerEnvV0):
    def __init__(self):
        self._sigma = 0.075
        super().__init__()
        # put frame_skip in sim object
        if self.frame_skip != 1:
            self.sim.nsubsteps = self.frame_skip
            self.frame_skip = 1

    def _step(self, a):
        ob, reward, done, res_dict = super()._step(a)

        if self.target_obj_sid != -1 and self.goal_sid != -1:
            target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
            goal_pos = self.data.site_xpos[self.goal_sid].ravel()

            res_dict['success'] = bool(np.linalg.norm(target_pos - goal_pos) < 0.010)
            # done = res_dict['success']
        return ob, reward, done, res_dict

    def state_vector(self):
        return self._get_obs()

    def _render(self, kargs, **kwargs):
        if 'close' not in kwargs or not kwargs['close']:
            self.mj_render()

    def reset_model(self):
        self.sim.reset()
        target_bid = self.model.body_name2id('nail_board')
        self.model.body_pos[target_bid, 2] = 0.175 + self.np_random.uniform(low=-self._sigma, high=self._sigma)
        self.sim.forward()
        return self._get_obs()

    def ss(self, state_dict, add_noise=False):
        """override to add noise to the door position based on sigma if required"""
        if add_noise:
            state_dict['board_pos'][2] += self.np_random.uniform(low=-self._sigma, high=self._sigma)
        return super().ss(state_dict)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value