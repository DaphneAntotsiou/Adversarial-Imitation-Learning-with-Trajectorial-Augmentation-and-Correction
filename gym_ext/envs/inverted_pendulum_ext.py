__author__ = 'DafniAntotsiou'
'''
This extends the InvertedPendulum-v2 gym environment 
'''

from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv


class InvertedPendulumEnvExt(InvertedPendulumEnv):
    def __init__(self):
        self._success = False
        self._sigma = 0.01      # sigma of noise to be added to actions
        self.name = None
        super().__init__()

    def reset(self):
        ret = super().reset()
        self._success = False
        return ret

    def step(self, a):
        ob, reward, done, step_dict = super().step(a)
        step_dict["success"] = not done
        self._success = step_dict["success"]
        if done:
            reward -= 1
        return ob, reward, done, step_dict

    def reset_model(self):
        # rewrite with sigma

        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-self._sigma, high=self._sigma)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-self._sigma, high=self._sigma)
        self.set_state(qpos, qvel)
        return self._get_obs()

    @property
    def success(self):
        return self._success

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    def gs(self):
        return dict(qpos=self.data.qpos.ravel().copy(), qvel=self.data.qvel.ravel().copy())

    def ss(self, state_dict, add_noise=False):
        if add_noise:
            state_dict['qpos'] += self.np_random.uniform(size=self.model.nq, low=-self._sigma, high=self._sigma)
            state_dict['qvel'] += self.np_random.randn(self.model.nv) * self._sigma

        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
