import numpy as np
from typing import List, Dict, Any
from planner.plan_result import PlanResult


def compute_cluster_metrics(hierarchical_plans: List[PlanResult]) -> Dict[str, Any]:
    """compute goal cluster reaching metrics for hierarchical planner"""
    reached_count = 0
    steps_to_cluster = []
    
    for plan in hierarchical_plans:
        cluster_traj = plan.cluster_trajectory
        goal_cluster = plan.goal_cluster_id
        
        # find first occurrence of goal cluster
        try:
            first_visit = cluster_traj.index(goal_cluster)
            reached_count += 1
            steps_to_cluster.append(first_visit)
        except (ValueError, AttributeError):
            # goal cluster never reached or cluster_trajectory is None
            pass
    
    num_plans = len(hierarchical_plans)
    reach_rate = reached_count / num_plans if num_plans > 0 else 0.0
    
    metrics = {
        'goal_cluster_reach_rate': reach_rate,
        'goal_cluster_reached_count': reached_count,
        'goal_cluster_failed_count': num_plans - reached_count,
    }
    
    if steps_to_cluster:
        metrics['avg_steps_to_goal_cluster'] = np.mean(steps_to_cluster)
        metrics['std_steps_to_goal_cluster'] = np.std(steps_to_cluster)
    else:
        metrics['avg_steps_to_goal_cluster'] = None
        metrics['std_steps_to_goal_cluster'] = None
    
    return metrics


def compute_success_metrics(plans: List[PlanResult], planner_name: str) -> Dict[str, Any]:
    """compute success rate and step statistics"""
    successful = [p for p in plans if p.success]
    failed = [p for p in plans if not p.success]
    
    num_plans = len(plans)
    success_count = len(successful)
    success_rate = success_count / num_plans if num_plans > 0 else 0.0
    
    metrics = {
        f'success_rate_{planner_name}': success_rate,
        f'success_count_{planner_name}': success_count,
        f'failure_count_{planner_name}': len(failed),
    }
    
    # successful plans statistics
    if successful:
        steps_success = [p.steps for p in successful]
        metrics[f'avg_steps_success_{planner_name}'] = np.mean(steps_success)
        metrics[f'std_steps_success_{planner_name}'] = np.std(steps_success)
    else:
        metrics[f'avg_steps_success_{planner_name}'] = None
        metrics[f'std_steps_success_{planner_name}'] = None
    
    # failed plans statistics
    if failed:
        distances_failed = [p.final_goal_distance for p in failed]
        metrics[f'avg_distance_failed_{planner_name}'] = np.mean(distances_failed)
    else:
        metrics[f'avg_distance_failed_{planner_name}'] = None
    
    return metrics