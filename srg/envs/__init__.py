from .registration import registry, register, make, spec

# Classic
# ----------------------------------------------------------------------
register(
    id='CartPole-v0',
    entry_point='srg.envs.classic:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,)

register(
    id='CartPole-v1',
    entry_point='srg.envs.classic:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,)

register(
    id='Pendulum-v0',
    entry_point='gym.envs.classic_control:PendulumEnv',
    max_episode_steps=200,
)