import numpy as np

from params import EvalArgs
from envs import RoomEnv
from utils import EnvironmentHelper


class PlannerConfig:
    """configuration parameters for planners"""
    def __init__(self, env: RoomEnv, env_helper: EnvironmentHelper, args: EvalArgs) -> None:
        # environment parameters
        self.batch_size = 1
        self.max_steps = env_helper.max_episode_steps
        self.goal_tolerance = env_helper.goal_tolerance
        
        # action constraints
        self.action_low = np.array(env.action_space.low)
        self.action_high = np.array(env.action_space.high)
        self.action_dim = env.action_space.shape[0]
        
        # config
        self.use_planner = args.use_planner
        self.use_prior_warmstart = args.use_prior_warmstart
        
        # planner optimization parameters
        self.horizon = args.horizon
        self.iterations = args.iterations
        self.samples = args.samples
        self.momentum = args.momentum
        self.sigma = args.sigma
        self.noise_beta = args.noise_beta

        # CEM specific parameters
        self.elite_ratio = args.elite_ratio

        # weight for cost terms
        self.distance_weight = args.distance_weight
        self.energy_weight =args.energy_weight
