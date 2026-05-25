import numpy as np
from typing import Union
import gymnasium as gym
from scipy.ndimage import gaussian_filter

from params import TrainArgs
from envs import RoomEnv, div_cast


class EnvironmentHelper:
    """helper for environment-specific operations"""
    def __init__(self, env: Union[RoomEnv, gym.Env], args: TrainArgs):
        self.env = env
        self.args = args
        self.env_type = args.env_type
        
        # rooms env specific attributes
        self.obs_type = self.args.obs_type

        # get room_size from RoomEnv if available
        if hasattr(env, 'room_size'):
            self.room_size = env.room_size
        
        # setup based on environment type
        if self.env_type == 'RoomEnv':
            self._setup_roomenv()
        elif self.env_type == 'OGBenchEnv':
            self._setup_ogbench()
        else:
            raise ValueError(f"{self.env_type} environment type is not supported yet, check again in some time!")
    
    # =========================================================================
    # setup envs
    # =========================================================================
    def _setup_roomenv(self):
        """setup RoomEnv-specific configuration"""
        self.state_dim = 2  # (x, y)
        self.state_shape = self.get_obs_shape()
        
        # action space
        self.action_dim = self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        # max episode steps and goal tolerance
        self.max_episode_steps = self.env.max_episode_steps
        self.goal_tolerance = self.env.goal_tolerance

    def _setup_ogbench(self):
        """setup OG-Bench-specific configuration"""
        obs_sample = self.env.observation_space.sample()
        self.state_dim = obs_sample.shape[0]
        self.state_shape = self.get_obs_shape()
        
        # action space
        self.action_dim = self.env.action_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        
        # cache observation dimensions for indexing
        self._achieved_goal_dim = 2 # (x, y) position

        # max episode steps and goal tolerance
        self.max_episode_steps = self.env.spec.max_episode_steps
        self.goal_tolerance = None  # ogbench uses info["success"] for success checking
    
    # =========================================================================
    # helper methods
    # =========================================================================
    def get_obs_shape(self):
        """get observation shape"""
        return self.env.observation_space.shape
    
    def reset_env(self, start_position=None, goal_position=None):
        """reset environment with optional start/goal states"""
        if self.env_type == 'RoomEnv':
            options = {}
            if start_position is not None:
                options['start_pos'] = start_position
            if goal_position is not None:
                options['goal_pos'] = goal_position
            
            obs, info = self.env.reset(options=options if options else None)
            return obs, info
        
        elif self.env_type == 'OGBenchEnv':
            # for OGBenchEnv, we can only reset using task_id
            raise NotImplementedError("use task_id for OGBenchEnv resets")
        
        else:
            raise ValueError(f"{self.env_type} environment type resetting is not supported yet, check again in some time!")
        
    def get_random_start_and_goal_position(self, start_position=None, goal_position=None):
        """get random start and goal position if not provided"""
        if start_position is None:
            if self.env_type == 'RoomEnv':
                start_position = div_cast(self.env.sample_feasible_position())
            elif self.env_type == 'OGBenchEnv':
                raise NotImplementedError("use task_id for OGBenchEnv resets")
            else:
                raise ValueError(f"{self.env_type} start state selection is not supported yet, check again in some time!")
        
        if goal_position is None:
            if self.env_type == 'RoomEnv':
                goal_position = div_cast(self.env.sample_feasible_position())
            elif self.env_type == 'OGBenchEnv':
                raise NotImplementedError("use task_id for OGBenchEnv resets")
            else:
                raise ValueError(f"{self.env_type} goal state selection is not supported yet, check again in some time!")

        return start_position, goal_position
    
    def extract_current_observation(self, obs):
        """extract current observation from environment output"""
        if self.env_type == 'RoomEnv' or self.env_type == 'OGBenchEnv':
            return np.array(obs)
        else:
            raise ValueError(f"{self.env_type} state extraction is not supported yet, check again in some time!")
        
    def get_obs_from_position(self, position: np.ndarray) -> np.ndarray:
        """
        convert position to observation
        For RoomEnv: converts xy position to observation (image or xy based on obs_type)
        """
        if self.env_type == 'RoomEnv':
            if self.obs_type == 'xy':
                return position
            elif self.obs_type == 'image':
                # convert position to image observation (same as RoomEnv._get_observation)
                obs = np.zeros((self.room_size[0], self.room_size[1], 1))
                pixel_x = int(np.clip(round(position[0] * self.room_size[0]), 0, self.room_size[0] - 1))
                pixel_y = int(np.clip(round(position[1] * self.room_size[1]), 0, self.room_size[1] - 1))

                # apply gaussian blur around agent
                obs[pixel_x, pixel_y, 0] = 1.0
                obs[:, :, 0] = gaussian_filter(obs[:, :, 0], sigma=1.0)
                if obs.max() > 0:
                    obs = obs / obs.max()
                return obs
            else:
                raise ValueError(f"Unknown obs_type: {self.obs_type}")
        elif self.env_type == 'OGBenchEnv':
            raise NotImplementedError("OGBenchEnv does not support position to observation conversion directly")
        else:
            raise ValueError(f"{self.env_type} state from position functionality is not supported yet, check again in some time!")
        
    def get_position_from_obs(self, state: np.ndarray) -> np.ndarray:
        """
        extract position from observation
        For RoomEnv: extracts xy position from observation (direct for xy, pixel search for image)
        """
        if self.env_type == 'RoomEnv':
            if self.obs_type == 'xy':
                return state
            elif self.obs_type == 'image':
                # find the pixel with highest value (use channel 0 explicitly)
                flat_idx = state[:, :, 0].argmax()

                # unravel to get (x, y) coordinates
                pixel_x, pixel_y = np.unravel_index(flat_idx, (self.room_size[0], self.room_size[1]))

                position = np.array([pixel_x / self.room_size[0], pixel_y / self.room_size[1]])
                return div_cast(position)
            else:
                raise ValueError(f"Unknown obs_type: {self.obs_type}")
        elif self.env_type == 'OGBenchEnv':
            # this assumes that the first two dimensions are (x, y) position
            return state[:self._achieved_goal_dim]
        else:
            raise ValueError(f"{self.env_type} position from state functionality is not supported yet, check again in some time!")
    
    def compute_goal_distance(self, current_state, goal_state):
        """compute Euclidean distance between current position and goal position"""
        current_position = self.get_position_from_obs(current_state)
        goal_position = self.get_position_from_obs(goal_state)

        return np.linalg.norm(current_position - goal_position)
