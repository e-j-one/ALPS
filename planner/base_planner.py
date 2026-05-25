import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Union
import gymnasium as gym

from params import EvalArgs
from envs import RoomEnv
from utils import EnvironmentHelper, normalize_obs
from allo import ALLOProcessor
from dynamics import Dynamics
from prior import Prior
from .config import PlannerConfig
from .optimizer import optimize_trajectory


class BasePlanner:
    """base class for all planners"""
    def __init__(self, env: Union[RoomEnv, gym.Env], env_helper: EnvironmentHelper, processor: ALLOProcessor, dynamics: Dynamics, prior: Prior, args: EvalArgs, key: jax.random.PRNGKey, save_dir=None):
        self.env = env
        self.env_type = processor.args.env_type
        self.env_helper = env_helper
        self.processor = processor
        self.dynamics = dynamics
        self.prior = prior
        self.args = args
        self.config = PlannerConfig(env, env_helper, args)
        self.rng_key = key
        self.save_dir = save_dir

    def get_random_start_and_goal_position(self, start_position=None, goal_position=None):
        """get some random start and goal position in case not specified"""
        return self.env_helper.get_random_start_and_goal_position(start_position, goal_position)
    
    def reset_env(self, start_position=None, goal_position=None):
        """reset environment based on type"""
        obs, info = self.env_helper.reset_env(start_position, goal_position)
        return obs, info
    
    def extract_current_observation(self, obs):
        """extract current obs from observation/info"""
        return self.env_helper.extract_current_observation(obs)
    
    def get_obs_from_position(self, position: np.ndarray) -> np.ndarray:
        """get full obs from position"""
        return self.env_helper.get_obs_from_position(position)
    
    def get_position_from_obs(self, obs: np.ndarray) -> np.ndarray:
        """get position from full obs"""
        return self.env_helper.get_position_from_obs(obs)
    
    def compute_goal_distance(self, current_obs, goal_obs):
        """compute distance between current and goal obss"""
        return self.env_helper.compute_goal_distance(current_obs, goal_obs)
    
    def observation_to_eigenspace(self, obs: np.ndarray) -> jnp.ndarray:
        """convert obs to eigenspace obs using ALLO"""
        return self.processor.observation_to_eigenspace(obs)

    def get_action(self, current_obs: jnp.ndarray, current_z: jnp.ndarray, target_z: jnp.ndarray) -> np.ndarray:
        """get optimal action using planner or prior"""
        if self.env_type == 'OGBenchEnv':
            # ogbench by default gives obs as np.uint8 in [0, 255] while RoomEnv gives float32 in [0, 1], hence normalize here
            current_obs = normalize_obs(current_obs)
        if self.args.use_planner:
            # use CEM with dynamics model
            horizon = self.config.horizon
            self.rng_key, subkey = jax.random.split(self.rng_key)

            optimal_action = optimize_trajectory(
                self.dynamics, self.prior, self.processor, current_obs,
                current_z, target_z, self.config, horizon, subkey,
            )

            return optimal_action
        else:
            # use prior only
            return np.array(self.prior(current_obs, current_z, target_z))
        
    def _save_video(self, filename: str):
        """save video of planning process"""
        import imageio
        
        video_path = os.path.join(self.save_dir, filename) if self.save_dir else filename
        return imageio.get_writer(video_path, fps=30, codec='libx264', quality=10, macro_block_size=1, pixelformat='yuv420p')
