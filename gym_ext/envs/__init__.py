__author__ = 'DafniAntotsiou'

from gym_ext.envs.inverted_pendulum_ext import InvertedPendulumEnvExt
from gym_ext.envs.HalfCheetah_ext import HalfCheetahEnvExt
from gym.envs.mujoco.mujoco_env import MujocoEnv
import mujoco_py
from mapping.mjviewerext import MjViewerExt as MjViewer


def _get_viewer(self, mode):
    self.viewer = self._viewers.get(mode)
    if self.viewer is None:
        if mode == 'human':
            self.viewer = MjViewer(self.sim, self.name if hasattr(self, 'name') else None)
        elif mode == 'rgb_array' or mode == 'depth_array':
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

        self.viewer_setup()
        self._viewers[mode] = self.viewer
    return self.viewer


setattr(MujocoEnv, '_get_viewer', _get_viewer)
