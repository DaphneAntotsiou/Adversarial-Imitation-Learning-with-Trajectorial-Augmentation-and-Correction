__author__ = 'DafniAntotsiou'

from gym.envs.registration import register


register(
    id='InvertedPendulum_ext-v2',
    entry_point='gym_ext.envs:InvertedPendulumEnvExt',
    max_episode_steps=1000,
)

register(
    id='HalfCheetah_ext-v2',
    entry_point='gym_ext.envs:HalfCheetahEnvExt',
    max_episode_steps=1000,
)
