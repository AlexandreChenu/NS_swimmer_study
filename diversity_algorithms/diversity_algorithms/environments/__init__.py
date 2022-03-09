# coding: utf-8

from diversity_algorithms.environments.gym_env import EvaluationFunctor

from diversity_algorithms.environments.dummy_env import SimpleMappingEvaluator

from diversity_algorithms.environments.environments import registered_environments

__all__=["gym_env", "behavior_descriptors", "environments", "dummy_env"]


from diversity_algorithms.environments.envs.swimmer_v3_det import SwimmerEnvDet

from gym.envs.registration import register

register(
    id='SwimmerDet-v3',
    entry_point='diversity_algorithms.environments.envs.swimmer_v3_det:SwimmerEnvDet',
)

register(
    id='BallInCup3dEnv-v0',
    entry_point='diversity_algorithms.environments.envs.ball_in_cup.ball_in_cup_3d_env:BallInCup3dEnv',
)
