__author__ = 'DafniAntotsiou'
'''
This extends the HalfCheetah-v2 gym environment 
'''

from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


class HalfCheetahEnvExt(HalfCheetahEnv):
    def __init__(self):
        self._sigma = 0.1       # sigma of noise to be added to actions
        self.traj_rew = 0       # total trajectory reward
        super().__init__()

    def reset_model(self):
        # rewrite with sigma
        self.traj_rew = 0

        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-self._sigma, high=self._sigma)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * self._sigma
        self.set_state(qpos, qvel)
        return self._get_obs()

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    def gs(self):
        return dict(qpos=self.data.qpos.ravel().copy(), qvel=self.data.qvel.ravel().copy(), traj_rew=self.traj_rew)

    def ss(self, state_dict, add_noise=False):
        if add_noise:
            state_dict['qpos'] += self.np_random.uniform(size=self.model.nq, low=-self._sigma, high=self._sigma)
            state_dict['qvel'] += self.np_random.randn(self.model.nv) * self._sigma

        if 'traj_rew' in state_dict:
            self.traj_rew = state_dict['traj_rew']
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)

    def step(self, action):
        ob, reward, done, res_dict = super().step(action)
        self.traj_rew += reward
        res_dict['success'] = max(self.traj_rew / 4463.46, 0)  # scale to gail paper
        return ob, reward, done, res_dict
