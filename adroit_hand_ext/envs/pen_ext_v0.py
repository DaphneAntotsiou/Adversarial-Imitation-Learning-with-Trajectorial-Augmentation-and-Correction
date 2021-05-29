__author__ = 'DafniAntotsiou'

"""
This script is an extension of PenEnvV0 from mj_envs
"""
from gym import error
try:
    from mujoco_py import MjSim
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

from mj_envs.hand_manipulation_suite.pen_v0 import PenEnvV0
import numpy as np
from mj_envs.utils.quatmath import euler2quat, quat2euler


class PenEnvExtV0(PenEnvV0):
    def __init__(self):
        self._sigma = 1.0
        super().__init__()
        # put frame_skip in sim object
        if self.frame_skip != 1:
            self.sim.nsubsteps = self.frame_skip
            self.frame_skip = 1

    def _step(self, a):
        ob, reward, done, res_dict = super()._step(a)

        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid]) / self.pen_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid]) / self.tar_length
        orien_similarity = np.dot(obj_orien, desired_orien)
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()

        res_dict['success'] = bool(orien_similarity > 0.97 and obj_pos[2] >= 0.15)
        # done = res_dict['success']
        return ob, reward, done, res_dict

    def state_vector(self):
        return self._get_obs()

    def _render(self, kargs, **kwargs):
        if 'close' not in kwargs or not kwargs['close']:
            self.mj_render()

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-self._sigma, high=self._sigma)
        desired_orien[1] = self.np_random.uniform(low=-self._sigma, high=self._sigma)
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.sim.forward()
        return self._get_obs()

    def ss(self, state_dict, add_noise=False):
        if add_noise:
            eul = quat2euler(state_dict['desired_orien'])
            eul[0] += self.np_random.uniform(low=-self._sigma, high=self._sigma)
            eul[1] += self.np_random.uniform(low=-self._sigma, high=self._sigma)
            state_dict['desired_orien'] = euler2quat(eul)
        return super().ss(state_dict)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
