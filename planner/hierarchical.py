import jax
import networkx as nx
from typing import Union
import gymnasium as gym

from params import EvalArgs
from envs import RoomEnv
from utils import EnvironmentHelper
from allo import ALLOProcessor
from dynamics import Dynamics
from prior import Prior
from clustering import SpectralClustering
from graph import ClusterGraph
from .base_planner import BasePlanner
from .plan_result import PlanResult


class HierarchicalPlanner(BasePlanner):
    def __init__(self, env: Union[RoomEnv, gym.Env], env_helper: EnvironmentHelper, processor: ALLOProcessor, dynamics: Dynamics, prior: Prior,
                 clustering: SpectralClustering, cluster_graph: ClusterGraph, args: EvalArgs, key: jax.random.PRNGKey, save_dir=None):
        super().__init__(env, env_helper, processor, dynamics, prior, args, key, save_dir)
        self.spectral_clustering = clustering
        self.save_dir = save_dir
        self.graph = cluster_graph
        self.args = args
    
    def plan(self, start_position=None, goal_position=None, task_id=None, eval_idx=None, record_video=False) -> PlanResult:
        """execute hierarchical planning"""
        # get video writer for storing planning process
        video_writer = None
        if record_video:
            if eval_idx is not None:
                video_filename = f"hierarchical_plan_task_{task_id}_eval_{eval_idx}.mp4"
            else:
                video_filename = f"hierarchical_plan_task_{task_id}.mp4"
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
            if start_position is None or goal_position is None:
                print("-"*60)
                if start_position is None:  
                    print("No start position provided, choosing a random start position")
                if goal_position is None:   
                    print("No goal position provided, choosing a random goal position")
                print("-"*60)
                
                # generate a random task if start and goal are not available
                start_position, goal_position = self.get_random_start_and_goal_position(start_position, goal_position)

            # reset environment
            obs, info = self.reset_env(start_position, goal_position)
            current_obs = self.extract_current_observation(obs)

            # get start and goal positions from environment
            start_position = info['start_pos']
            goal_position = info['goal_pos']
            goal_obs = self.get_obs_from_position(goal_position)

        print(f"Hierarchical planning from {start_position} to {goal_position}")
        print("-"*60)

        # record initial frame
        if video_writer:
            frame = self.env.render()
            if frame is not None: video_writer.append_data(frame)

        success = False

        # get cluster IDs for the corresponding observations
        start_cluster_id = self.spectral_clustering.predict_cluster_for_observation(current_obs)
        goal_cluster_id = self.spectral_clustering.predict_cluster_for_observation(goal_obs)
        
        # compute high-level cluster path using Dijkstra
        cluster_path = self.compute_cluster_path(start_cluster_id, goal_cluster_id)
        
        current_z = self.observation_to_eigenspace(current_obs)
        goal_z = self.observation_to_eigenspace(goal_obs)
        
        # initialize cluster tracking
        cluster_trajectory = []
        
        # hierarchical execution loop
        cluster_path_index = 0
        current_target_cluster = cluster_path[min(cluster_path_index + 1, len(cluster_path) - 1)]
        
        for step in range(self.config.max_steps):
            # determine current cluster
            current_cluster = self.spectral_clustering.predict_cluster_for_observation(current_obs)
            cluster_trajectory.append(current_cluster)

            # check if current cluster is still in path
            if current_cluster not in cluster_path:
                print(f"=== Recomputing path from cluster {current_cluster} to goal cluster {goal_cluster_id} ===")
                # recompute path from current position to goal
                cluster_path = self.compute_cluster_path(current_cluster, goal_cluster_id)
                cluster_path_index = 0
                current_target_cluster = cluster_path[min(cluster_path_index + 1, len(cluster_path) - 1)]

            # update cluster path if reached current target
            elif current_cluster == current_target_cluster and cluster_path_index < len(cluster_path) - 1:
                cluster_path_index += 1
                current_target_cluster = cluster_path[min(cluster_path_index + 1, len(cluster_path) - 1)]
                print(f"=== Step {step}: Reached cluster {current_cluster}, next {current_target_cluster} ===")
            
            # set target eigenspace representation
            if current_target_cluster == goal_cluster_id:
                target_z = goal_z
            else:
                target_z = self.spectral_clustering.eigenspace_centroids[current_target_cluster]
            
            # get action from planner or prior
            executed_action = self.get_action(current_obs, current_z, target_z)
            
            # step environment
            next_obs, _, _, _, next_info = self.env.step(executed_action)
            next_obs= self.extract_current_observation(next_obs)
            next_z = self.observation_to_eigenspace(next_obs)
            
            # record video frame
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
            
            # if success, record final cluster and break
            if success:
                final_cluster = self.spectral_clustering.predict_cluster_for_observation(current_obs)
                cluster_trajectory.append(final_cluster)
                break
        
        if success:
            goal_distance = self.compute_goal_distance(current_obs, goal_obs)
            print(f"Hierarchical planner succeeded in reaching the goal in {step + 1} steps!")
        else:
            goal_distance = self.compute_goal_distance(current_obs, goal_obs)
            print(f"Hierarchical planner failed to reach goal within max steps, Final distance to goal: {goal_distance:.3f}")
        
        # close video writer
        if video_writer:
            video_writer.close()

        return PlanResult(start_position, goal_position, step+1, goal_distance, success, 'hierarchical', start_cluster_id, goal_cluster_id, cluster_path, cluster_trajectory)
    
    def compute_cluster_path(self, start_cluster, goal_cluster):
        """compute shortest path between clusters"""
        graph = self.graph.G
        try:
            path = nx.shortest_path(graph, start_cluster, goal_cluster)
            print(f"Cluster path: {path}")
            return path
        except nx.NetworkXNoPath:
            print(f"No path found between cluster {start_cluster} and {goal_cluster}, returning direct path")
            return [start_cluster, goal_cluster]
