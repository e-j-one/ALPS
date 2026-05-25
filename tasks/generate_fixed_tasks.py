import os
import json
import tyro
import numpy as np
from copy import deepcopy

from envs import setup_environment
from params import TrainArgs
from utils.env_helpers import EnvironmentHelper


# run command:
# python -m tasks.generate_fixed_tasks --env-type RoomEnv --num-tasks 50 --save-dir ./tasks

def generate_tasks(env_name: str, args: TrainArgs, save_dir="./tasks"):
    """generate tasks for a given environment"""
    num_tasks = args.num_tasks
    print(f"Generating {num_tasks} fixed tasks for {env_name}...")
    
    # setup environment
    env_args = deepcopy(args)
    env_args.env_name = env_name
    
    env = setup_environment(env_args)
    
    # create environment helper
    env_helper = EnvironmentHelper(env, env_args)
    
    # set seed for reproducibility
    np.random.seed(14)
    if hasattr(env, 'seed'):
        env.seed(14)
    elif hasattr(env, 'reset'):
        env.reset(seed=14)
    
    tasks = []
    for task_id in range(num_tasks):
        # get random start and goal positions
        start_pos, goal_pos = env_helper.get_random_start_and_goal_position()
        
        start_pos = np.array(start_pos)
        goal_pos = np.array(goal_pos)
        
        task = {
            'task_id': task_id,
            'start_position': start_pos.tolist(),
            'goal_position': goal_pos.tolist(),
        }
        tasks.append(task)
    
    # save tasks
    os.makedirs(save_dir, exist_ok=True)
    task_file = os.path.join(save_dir, f"{env_name}_tasks.json")
    
    task_data = {
        'env_name': env_name,
        'env_type': env_args.env_type,
        'num_tasks': num_tasks,
        'generation_seed': 14,
        'tasks': tasks
    }
    
    with open(task_file, 'w') as f:
        json.dump(task_data, f, indent=2)
    
    print(f"Saved {num_tasks} tasks to {task_file}")
    
    return task_file


def main():
    """generate fixed tasks for all environments"""
    args = tyro.cli(TrainArgs)
    
    if args.env_type == 'RoomEnv':
        envs = ['rooms', 'hallway', 'spiral']
    else:
        print(f"Unknown env_type: {args.env_type}")
        return
    
    print(f"Generating fixed tasks for {args.env_type} environments:")
    
    task_files = []
    for env_name in envs:
        print(f"\n[{env_name.upper()}]")
        task_file = generate_tasks(env_name, args)
        task_files.append(task_file)
    
    print(f"\nGenerated {len(task_files)} task files!")


if __name__ == "__main__":
    main()