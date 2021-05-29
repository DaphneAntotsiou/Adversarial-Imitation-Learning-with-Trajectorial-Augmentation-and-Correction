__author__ = 'DafniAntotsiou'

from gym.envs.registration import register


register(
    id='door_ext-v0',
    entry_point='adroit_hand_ext.envs:DoorEnvExtV0',
    max_episode_steps=100,
)

register(
    id='door_ext_200-v0',
    entry_point='adroit_hand_ext.envs:DoorEnvExtV0',
    max_episode_steps=200,
)

register(
    id='pen_ext-v0',
    entry_point='adroit_hand_ext.envs:PenEnvExtV0',
    max_episode_steps=200,
)

register(
    id='hammer_ext-v0',
    entry_point='adroit_hand_ext.envs:HammerEnvExtV0',
    max_episode_steps=200,
)

register(
    id='hammer_ext_500-v0',
    entry_point='adroit_hand_ext.envs:HammerEnvExtV0',
    max_episode_steps=500,
)