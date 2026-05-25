import ogbench

from .room_envs import create_room_environment
from params import TrainArgs


def setup_rooms_environment(args: TrainArgs, render_mode=None):
    """setup and configure the rooms environment"""
    print("Setting up rooms environment...")
    env_name = args.rooms_env_name
    env = create_room_environment(env_name, room_size=(args.room_size, args.room_size), max_episode_steps=args.max_episode_steps, obs_type=args.obs_type, render_mode=render_mode)
    env.reset(seed=args.seed)
    
    print(f"Environment created: {env_name}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    return env


def setup_ogbench_environment(args: TrainArgs, render_mode=None):
    """setup and configure the ogbench environment"""
    print("Setting up OG-Bench environment...")
    task = args.ogbench_task_name
    env_kwargs = {}
    if render_mode is not None:
        env_kwargs['render_mode'] = render_mode
    if args.obs_type != "image":
        env_kwargs['width'] = 480
        env_kwargs['height'] = 480
    env = ogbench.make_env_and_datasets(task, env_only=True, **env_kwargs)

    env.reset(seed=args.seed)

    print(f"Environment created: {task}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    return env


def setup_environment(args: TrainArgs, render_mode=None):
    """setup gym environment"""
    if args.env_type == 'RoomEnv':
        return setup_rooms_environment(args, render_mode)
    elif args.env_type == 'OGBenchEnv':
        return setup_ogbench_environment(args, render_mode)
    else:
        raise ValueError(f"Unknown env_type: {args.env_type}")
