from typing import Tuple, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from shapely.geometry import Point, LineString, Polygon
from scipy.ndimage import gaussian_filter


def div_cast(x, m=100):
    """round coordinates to specified precision"""
    if isinstance(x, (list, np.ndarray)):
        return np.array([round(val * m) / m for val in x])
    else:
        return round(x * m) / m


class RoomEnv(gym.Env):
    """continuous 2D room environment with walls and passages"""
    def __init__(self, room_size: Tuple[int, int] = (100, 100), wall_thickness: float = 0.009, 
                 reward_type: str = 'sparse',  # 'sparse' or 'dense'
                 max_episode_steps: int = 30,
                 obs_type: str = 'image',  # 'image' or 'xy'
                 render_mode: Optional[str] = None # 'human' or 'rgb_array' or None
                ):
        
        super().__init__()
        # initialize random number generator
        self.np_random, _ = gym.utils.seeding.np_random(None)
        
        # environment parameters
        self.room_size = room_size
        self.wall_thickness = wall_thickness
        self.reward_type = reward_type
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = 0.03
        self.obs_type = obs_type
        self.current_step = 0
        self.render_mode = render_mode
        
        # action space
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(2,), dtype=np.float64)
        
        # observation space based on type
        if obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=1, shape=(room_size[0], room_size[1], 1), dtype=np.float64)
        elif obs_type == 'xy':
            self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float64)
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")
        
        # set up walls and obstacles
        self._setup_walls()
        
        # sample initial start and goal positions
        self.start_pos = None
        self.goal_pos = None
        self.agent_pos = None
        
        # reward tracking for dense rewards
        self.reached_01_threshold = False  # d_g < 0.1
        self.reached_005_threshold = False  # d_g < 0.05
    
    def _setup_walls(self):
        """set up wall obstacles in the environment - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _setup_walls")
    
    def _is_position_feasible(self, pos: np.ndarray, buffer_distance: float = 0.01) -> bool:
        """check if a position is feasible (not inside walls with buffer)"""
        x, y = pos
        
        # check boundaries with buffer
        if x < buffer_distance or x > (1.0 - buffer_distance):
            return False
        if y < buffer_distance or y > (1.0 - buffer_distance):
            return False
        
        # check collision with obstacles
        point = Point(x, y)
        
        for obstacle in self.obstacles:
            # add buffer around obstacles
            buffered_obstacle = obstacle.buffer(buffer_distance)
            if buffered_obstacle.contains(point):
                return False
        
        return True
    
    def sample_feasible_position(self, max_attempts: int = 100) -> np.ndarray:
        """sample a random feasible position in the environment"""
        for _ in range(max_attempts):
            # sample random position with some margin from boundaries
            pos = self.np_random.uniform(0.01, 0.99, 2)
            
            if self._is_position_feasible(pos):
                return pos
        
        # fallback to a known safe position if sampling fails
        print("Warning: Could not find feasible position, using fallback")
        return np.array([0.1, 0.1])
    
    def _sample_start_goal_positions(self, min_distance: float = 0.3) -> None:
        """sample start and goal positions ensuring minimum distance"""
        max_attempts = 100
        
        # sample start position
        self.start_pos = div_cast(self.sample_feasible_position())
        
        # sample goal position with minimum distance constraint
        for _ in range(max_attempts):
            goal_candidate = self.sample_feasible_position()
            if np.linalg.norm(self.start_pos - goal_candidate) >= min_distance:
                self.goal_pos = div_cast(goal_candidate)
                return
            
        # fallback if no valid goal found
        print("Warning: Could not find valid goal position, using fallback")
        self.goal_pos = div_cast(self.sample_feasible_position())
    
    def _create_wall_polygon(self, start_pos: Tuple[float, float], 
                           end_pos: Tuple[float, float], 
                           wall_type: str) -> Tuple[Polygon, str]:
        """create a wall polygon with specified thickness"""
        x1, y1 = start_pos
        x2, y2 = end_pos
        delta = self.wall_thickness
        
        if wall_type == 'vertical':
            polygon = Polygon([
                (x1 - delta, y1 - delta), 
                (x1 - delta, y2 + delta),
                (x1 + delta, y2 + delta), 
                (x1 + delta, y1 - delta)
            ])
        elif wall_type == 'horizontal':
            polygon = Polygon([
                (x1 - delta, y1 - delta), 
                (x1 - delta, y1 + delta),
                (x2 + delta, y1 + delta), 
                (x2 + delta, y1 - delta)
            ])
        else:
            raise ValueError(f"Unknown wall type: {wall_type}")
            
        return polygon, wall_type
    
    def _detect_collision(self, start_pos: np.ndarray, end_pos: np.ndarray) -> np.ndarray:
        """get closest intersection point"""
        if np.allclose(start_pos, end_pos):
            return div_cast(end_pos)
            
        agent_ray = LineString([(start_pos[0], start_pos[1]), (end_pos[0], end_pos[1])])
        
        intersections = []
        
        for obstacle in self.obstacles:
            intersect = obstacle.intersection(agent_ray)
            
            # detect if there's an intersection
            if not 'EMPTY' in str(intersect):
                intersect_points_list = [np.array(item) for item in list(intersect.coords)]
                intersections.extend(intersect_points_list)
        
        if len(intersections) == 0:
            return div_cast(end_pos)  # no collision, move to end position
        else:
            # find closest intersection to start position
            distances = [np.linalg.norm(item - start_pos, 2) for item in intersections]
            closest_idx = np.argmin(distances)
            closest_intersection = intersections[closest_idx]
            
            return div_cast(closest_intersection)
    
    def _clip_to_bounds(self, pos: np.ndarray) -> np.ndarray:
        """clip position to environment bounds"""
        x, y = pos
        if x <= 0: x = 0.0
        if x >= 1: x = 0.99
        
        if y <= 0: y = 0.0
        if y >= 1: y = 0.99
        
        return np.array([x, y])
    
    def _get_observation(self) -> np.ndarray:
        """generate observation based on observation type"""
        if self.obs_type == 'image':
            obs = np.zeros((self.room_size[0], self.room_size[1], 1))
            
            # convert continuous position to pixel coordinates
            pixel_x = int(np.clip(round(self.agent_pos[0] * self.room_size[0]), 0, self.room_size[0] - 1))
            pixel_y = int(np.clip(round(self.agent_pos[1] * self.room_size[1]), 0, self.room_size[1] - 1))
            
            # set agent position in image
            obs[pixel_x, pixel_y, 0] = 1.0
                    
            # apply gaussian blur around agent
            obs[:, :, 0] = gaussian_filter(obs[:, :, 0], sigma=1.0)
            obs = obs / obs.max() if obs.max() > 0 else obs

            return obs
            
        elif self.obs_type == 'xy':
            return self.agent_pos.copy()
        
        else:
            raise ValueError(f"Unknown observation type: {self.obs_type}")
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """calculate reward based on reward type"""
        goal_distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        if self.reward_type == 'sparse':
            return 1.0 if goal_distance < self.goal_tolerance else 0.0
                
        elif self.reward_type == 'dense':
            # dense reward with first-time thresholds
            reward = 0.0
            
            if goal_distance < self.goal_tolerance:
                reward = 1.0
            elif goal_distance < 0.05 and not self.reached_005_threshold:
                reward = 0.5
                self.reached_005_threshold = True
            elif goal_distance < 0.1 and not self.reached_01_threshold:
                reward = 0.25
                self.reached_01_threshold = True
                
            return reward
        
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def _is_terminal(self) -> bool:
        """check if episode should terminate"""
        goal_distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        return goal_distance < self.goal_tolerance or self.current_step >= self.max_episode_steps
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """execute one step in the environment"""
        # clip action to valid range
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -0.2, 0.2)
        
        # store previous position
        prev_pos = self.agent_pos.copy()
        
        # calculate new position
        new_pos = self.agent_pos + action
        new_pos = self._clip_to_bounds(new_pos)
        
        # check for collisions and update position
        self.agent_pos = self._detect_collision(prev_pos, new_pos)
        self.agent_pos = div_cast(self.agent_pos)
        
        # calculate reward
        reward = self._calculate_reward(action)
        
        # check termination
        terminated = self._is_terminal()
        truncated = self.current_step >= self.max_episode_steps
        
        # generate observation
        obs = self._get_observation()
        
        # update step counter
        self.current_step += 1
        
        # info
        info = {
            'agent_pos': self.agent_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'goal_distance': np.linalg.norm(self.agent_pos - self.goal_pos),
            'step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """reset the environment"""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            self.action_space.seed(seed)
            if hasattr(self.observation_space, 'seed'):
                self.observation_space.seed(seed)

        # handle start/goal position options
        if options:
            if 'start_pos' in options:
                self.start_pos = div_cast(np.array(options['start_pos']))
            if 'goal_pos' in options:
                self.goal_pos = div_cast(np.array(options['goal_pos']))
            
            # sample missing positions
            if 'start_pos' not in options and 'goal_pos' not in options:
                self._sample_start_goal_positions()
            elif 'start_pos' not in options:
                self.start_pos = div_cast(self.sample_feasible_position())
            elif 'goal_pos' not in options:
                self.goal_pos = div_cast(self.sample_feasible_position())
        else:
            self._sample_start_goal_positions()
        
        # reset agent position to start
        self.agent_pos = self.start_pos.copy()
        
        # reset counters and flags
        self.current_step = 0
        self.reached_01_threshold = False
        self.reached_005_threshold = False
        
        # generate initial observation
        obs = self._get_observation()
        
        info = {
            'agent_pos': self.agent_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'start_pos': self.start_pos.copy(),
            'goal_distance': np.linalg.norm(self.agent_pos - self.goal_pos)
        }
        
        return obs, info
    
    def render(self) -> Optional[np.ndarray]:
        """render the environment"""
        if self.render_mode is None:
            return None
        
        plt.figure(figsize=(8, 8))
        
        # draw walls
        for obstacle in self.obstacles:
            x, y = obstacle.exterior.xy
            plt.plot(x, y, 'k-', linewidth=3)
            plt.fill(x, y, color='gray', alpha=0.7)
        
        # draw agent, start, and goal
        plt.plot(self.agent_pos[0], self.agent_pos[1], 'ro', markersize=10, label='Agent')
        plt.plot(self.start_pos[0], self.start_pos[1], 'bs', markersize=14, label='Start')
        plt.plot(self.goal_pos[0], self.goal_pos[1], 'g*', markersize=14, label='Goal')
        
        # set limits and labels
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(f'{self.__class__.__name__} - Step {self.current_step}')
        
        if self.render_mode == 'human':
            plt.show(block=False)
            plt.pause(0.01)
            return None
        elif self.render_mode == 'rgb_array':
            fig = plt.gcf()
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)
            img = img[:, :, :3]  # remove alpha channel
            plt.close()
            return img
        else:
            plt.close()
            return None
    
    def close(self):
        """close the environment and clean up resources"""
        plt.close('all')

    def seed(self, seed=None):
        """seed the env's random number generator"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.action_space.seed(seed)
        if hasattr(self.observation_space, 'seed'):
            self.observation_space.seed(seed)
        
        return [seed]