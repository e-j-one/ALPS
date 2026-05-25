import numpy as np
import jax
import jax.numpy as jnp
from typing import Union
import gymnasium as gym

from params import EvalArgs
from envs import RoomEnv
from utils import EnvironmentHelper
from allo import ALLOProcessor
from dynamics import Dynamics
from prior import Prior
from .base_planner import BasePlanner
from .plan_result import PlanResult


class CEMPlanner(BasePlanner):
    def __init__(self, env: Union[RoomEnv, gym.Env], env_helper: EnvironmentHelper, processor: ALLOProcessor, dynamics: Dynamics, prior: Prior, args:EvalArgs, key: jax.random.PRNGKey, save_dir=None):
        super().__init__(env, env_helper, processor, dynamics, prior, args, key, save_dir)
    
    def plan(self, start_position=None, goal_position=None, task_id=None, eval_idx=None, record_video=False) -> PlanResult:
        """execute pure CEM/MPPI planning"""
        # get video writer for storing planning process
        video_writer = None
        if record_video:
            if eval_idx is not None:
                video_filename = f"cem_plan_task_{task_id}_eval_{eval_idx}.mp4"
            else:
                video_filename = f"cem_plan_task_{task_id}.mp4"
            video_writer = self._save_video(video_filename)
        
        # get start and goal observations
        if self.env_type == 'OGBenchEnv':
            if task_id is None: raise ValueError("OGBenchEnv requires a task_id")
            obs, info = self.env.reset(options=dict(task_id=task_id))
            current_obs = self.extract_current_observation(obs)
            goal_obs = info['goal']
            
            # get positions just for the PlanResult and logging
            start_position = self.get_position_from_obs(current_obs)
            goal_position = self.get_position_from_obs(goal_obs)

        else:
            # for RoomEnv
            if start_position is None or goal_position is None:
                print("-"*60)
                if start_position is None:  
                    print("No start position provided, choosing a random start position")
                if goal_position is None:   
                    print("No goal position provided, choosing a random goal position")
                print("-"*60)
                
                # generate a random task if start and goal are not available
                start_position, goal_position = self.get_random_start_and_goal_position(start_position, goal_position)

            obs, info = self.reset_env(start_position, goal_position)
            current_obs = self.extract_current_observation(obs)

            # get start and goal positions from environment
            start_position = info['start_pos']
            goal_position = info['goal_pos']
            goal_obs = self.get_obs_from_position(goal_position)

        print(f"CEM planning from {start_position} to {goal_position}")
        print("-"*60)

        # record initial frame
        if video_writer:
            frame = self.env.render()
            if frame is not None: video_writer.append_data(frame)
        
        success = False

        # get eigenspace projections
        current_z = self.observation_to_eigenspace(current_obs)
        goal_z = self.observation_to_eigenspace(goal_obs)
        
        # CEM/MPPI Planning Loop
        for step in range(self.config.max_steps):
            # get action from planner or prior
            executed_action = self.get_action(current_obs, current_z, goal_z)

            # step environment
            next_obs, _, _, _, next_info = self.env.step(executed_action)
            next_obs = self.extract_current_observation(next_obs)
            next_z = self.observation_to_eigenspace(next_obs)
            
            # render planning process
            if video_writer:
                frame = self.env.render()
                if frame is not None: video_writer.append_data(frame)
            
            # update observation and eigenspace projection
            current_obs = next_obs
            current_z = next_z
            
            # check if goal is reached
            if self.env_type == 'OGBenchEnv':
                # OGBench provides a clear success flag in the info dict
                if next_info.get('success', False):
                    success = True
            else:
                # for RoomEnv
                goal_distance = self.compute_goal_distance(current_obs, goal_obs)
                if goal_distance < self.config.goal_tolerance:
                    success = True

            if success:
                break

        # close video writer
        if video_writer:
            video_writer.close()
        
        if success:
            goal_distance = self.compute_goal_distance(current_obs, goal_obs)
            print(f"CEM planner succeeded in reaching the goal in {step + 1} steps!")
        else:
            goal_distance = self.compute_goal_distance(current_obs, goal_obs)
            print(f"CEM planner failed to reach goal within max steps, Final distance to goal: {goal_distance:.3f}")

        return PlanResult(start_position, goal_position, step+1, goal_distance, success, 'cem')
