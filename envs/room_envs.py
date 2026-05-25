from .base_env import RoomEnv


class RoomsEnv(RoomEnv):
    """room environment with parallel rooms"""
    def _setup_walls(self):
        """set up wall obstacles for rooms environment"""
        wall_specs = [
            [(0.25, 0.00), (0.25, 0.80), 'vertical'],  # Wall with gap at top
            [(0.25, 0.90), (0.25, 1.00), 'vertical'],  # Top segment
            [(0.50, 0.00), (0.50, 0.20), 'vertical'],  # Wall with gap in middle
            [(0.50, 0.30), (0.50, 1.00), 'vertical'],  # Top segment
            [(0.75, 0.00), (0.75, 0.45), 'vertical'],  # Wall with gap in middle
            [(0.75, 0.55), (0.75, 1.00), 'vertical'],  # Top segment
        ]
        
        self.obstacles = []
        self.obstacle_types = []
        
        for wall_def in wall_specs:
            start_pos, end_pos, wall_type = wall_def
            obstacle, obs_type = self._create_wall_polygon(start_pos, end_pos, wall_type)
            self.obstacles.append(obstacle)
            self.obstacle_types.append(obs_type)


class HallwayEnv(RoomEnv):
    """room environment like a hallway forming a cross pattern"""
    def _setup_walls(self):
        """set up wall obstacles for hallway environment"""
        wall_specs = [
            [(0.50, 0.00), (0.50, 0.40), 'vertical'],   # Vertical wall bottom
            [(0.50, 0.60), (0.50, 1.00), 'vertical'],   # Vertical wall top
            [(0.20, 0.40), (0.80, 0.40), 'horizontal'], # Horizontal wall bottom
            [(0.20, 0.60), (0.80, 0.60), 'horizontal'], # Horizontal wall top
        ]
        
        self.obstacles = []
        self.obstacle_types = []
        
        for wall_def in wall_specs:
            start_pos, end_pos, wall_type = wall_def
            obstacle, obs_type = self._create_wall_polygon(start_pos, end_pos, wall_type)
            self.obstacles.append(obstacle)
            self.obstacle_types.append(obs_type)


class SpiralEnv(RoomEnv):
    """room environment with spiral-shaped obstacles"""
    def _setup_walls(self):
        """set up wall obstacles for spiral environment"""
        wall_specs = [
            [(0.20, 0.00), (0.20, 0.75), 'vertical'],   # Left outer wall
            [(0.20, 0.75), (0.80, 0.75), 'horizontal'], # Top outer wall
            [(0.80, 0.25), (0.80, 0.75), 'vertical'],   # Right outer wall
            [(0.40, 0.25), (0.80, 0.25), 'horizontal'], # Bottom inner wall
            [(0.40, 0.25), (0.40, 0.50), 'vertical'],   # Left inner wall
            [(0.40, 0.50), (0.60, 0.50), 'horizontal'], # Top inner wall
        ]
        
        self.obstacles = []
        self.obstacle_types = []
        
        for wall_def in wall_specs:
            start_pos, end_pos, wall_type = wall_def
            obstacle, obs_type = self._create_wall_polygon(start_pos, end_pos, wall_type)
            self.obstacles.append(obstacle)
            self.obstacle_types.append(obs_type)


class OpenRoomEnv(RoomEnv):
    """open room environment with no walls"""
    def _setup_walls(self):
        """no walls in open room environment"""
        self.obstacles = []
        self.obstacle_types = []


def create_room_environment(env_name='rooms', room_size=(100, 100), reward_type='sparse', max_episode_steps=30, obs_type='image', render_mode=None):
    """function to create the room environment"""
    env_classes = {
        'rooms': RoomsEnv,
        'hallway': HallwayEnv,
        'spiral': SpiralEnv,
        'openroom': OpenRoomEnv
    }
    
    if env_name not in env_classes:
        raise ValueError(f"Unknown environment type: {env_name}. Available: {list(env_classes.keys())}")
    
    env_class = env_classes[env_name]
    env = env_class(
        room_size=room_size,
        reward_type=reward_type,
        max_episode_steps=max_episode_steps,
        obs_type=obs_type,
        render_mode=render_mode
    )
    return env
