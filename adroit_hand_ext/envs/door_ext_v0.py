__author__ = 'DafniAntotsiou'

"""
This script is an extension of DoorEnvV0 from mj_envs
"""

from mj_envs.hand_manipulation_suite.door_v0 import DoorEnvV0


class DoorEnvExtV0(DoorEnvV0):
    def __init__(self):
        self._sigma = 0.05
        self._success = False
        super().__init__()
        # put frame_skip in sim object
        if self.frame_skip != 1:
            self.sim.nsubsteps = self.frame_skip
            self.frame_skip = 1

    def _step(self, a):
        ob, reward, done, res_dict = super()._step(a)

        door_pos = self.data.qpos[self.door_hinge_did]

        res_dict['success'] = bool(door_pos > 1.2)      # current frame
        # res_dict['success'] = bool(door_pos > 1.0)      # current frame

        self._success = res_dict['success'] or self._success    # entire trajectory

        return ob, reward, done, res_dict

    def state_vector(self):
        return self._get_obs()

    def _render(self, kargs, **kwargs):
        self.mj_render()

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        # add sigma in door initialisation
        self.model.body_pos[self.door_bid, 0] = -0.25 + self.np_random.uniform(low=-self._sigma, high=self._sigma)
        self.model.body_pos[self.door_bid, 1] = 0.3 + self.np_random.uniform(low=-self._sigma, high=self._sigma)
        self.model.body_pos[self.door_bid, 2] = 0.3 + self.np_random.uniform(low=-self._sigma, high=self._sigma)

        self.sim.forward()
        ret = self._get_obs()

        self._success = False
        return ret

    def ss(self, state_dict, add_noise=False):
        """override to add noise to the door position based on sigma if required"""
        if add_noise:
            state_dict['door_body_pos'] += self.np_random.uniform(low=-self._sigma, high=self._sigma, size=3)
        return super().ss(state_dict)

    @property
    def success(self):
        return self._success

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
