import os
import json
import numpy as np
import jax
from typing import Dict, Any, List, Union

from params import TrainArgs, EvalArgs
from planner import Planner
from planner.plan_result import PlanResult
from .metrics import compute_cluster_metrics, compute_success_metrics


def load_tasks(env_name: str, task_dir: str = "./tasks"):
    """load pre-generated fixed tasks"""
    task_file = os.path.join(task_dir, f"{env_name}_tasks.json")
    
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"Fixed task file not found: {task_file}")
    
    with open(task_file, 'r') as f:
        task_data = json.load(f)
    
    print(f"Loaded {len(task_data['tasks'])} fixed tasks from {task_file}")
    return task_data['tasks']


def evaluate_planners(planner: Planner, args: Union[TrainArgs, EvalArgs], save_dir: str = None):
    """evaluate both hierarchical and CEM planners on fixed tasks"""
    if args.env_type == 'RoomEnv':
        env_name = args.rooms_env_name
    elif args.env_type == 'OGBenchEnv':
        env_name = args.ogbench_task_name
    else:
        raise ValueError(f"Unsupported environment type: {args.env_type}")

    if args.env_type == 'OGBenchEnv':
        print(f"Generating OGBench task list ...")
        tasks = []
        num_ogbench_tasks = 5
        num_evals_per_task = 50

        for task_id in range(1, num_ogbench_tasks + 1):
            for eval_idx in range(num_evals_per_task):
                tasks.append({ 'task_id': task_id, 'eval_idx': eval_idx })

        print(f"Generated {len(tasks)} OGBench evaluations ({num_evals_per_task} evals x {num_ogbench_tasks} tasks).")
    
    else:
        # for RoomEnv
        tasks = load_tasks(env_name, args.tasks_dir)

    num_tasks = len(tasks)
    
    print(f"\n{'='*80}")
    print(f"PLANNING EVALUATION")
    print(f"{'='*80}")
    print(f"Number of tasks: {num_tasks}")
    print(f"Number of clusters: {args.num_clusters}")
    print(f"Goal tolerance: {args.goal_tolerance}")
    print(f"{'='*80}\n")
    
    # get planners
    hierarchical_planner = planner.get_hierarchical_planner()
    cem_planner = planner.get_cem_planner() if args.eval_cem_planner else None
    
    # store results
    hierarchical_plans: List[PlanResult] = []
    cem_plans: List[PlanResult] = []
    failed_ids_hierarchical = []
    failed_ids_cem = []

    # run evaluation
    for i, task in enumerate(tasks):
        task_id = task['task_id']

        if args.env_type == 'OGBenchEnv':
            start_position = None
            goal_position = None
            eval_idx = task.get('eval_idx', 0)
            print(f"\nTask {task_id}, Eval {eval_idx + 1}/50: Start {start_position} -> Goal {goal_position}")
        else:
            start_position = np.array(task['start_position'])
            goal_position = np.array(task['goal_position'])
            eval_idx = None
            print(f"\nTask {task_id}: Start {start_position} -> Goal {goal_position}")

        # render first 5 evaluations only
        record_video = args.render and (i < 5)
        
        # === HIERARCHICAL PLANNER ===
        h_plan = hierarchical_planner.plan(start_position, goal_position, task_id=task_id, eval_idx=eval_idx, record_video=record_video)
        hierarchical_plans.append(h_plan)
        if not h_plan.success:
            failed_ids_hierarchical.append((task_id, eval_idx))

        # === CEM PLANNER ===
        if cem_planner is not None:
            c_plan = cem_planner.plan(start_position, goal_position, task_id=task_id, eval_idx=eval_idx, record_video=record_video)
            cem_plans.append(c_plan)
            if not c_plan.success:
                failed_ids_cem.append((task_id, eval_idx))

        # clear JAX caches to avoid memory issues
        jax.clear_caches()
        
        print("\n" + "="*50)
    
    # compute metrics
    h_metrics = compute_success_metrics(hierarchical_plans, 'hierarchical')
    c_metrics = compute_success_metrics(cem_plans, 'cem') if cem_planner is not None else {}
    cluster_metrics = compute_cluster_metrics(hierarchical_plans)
    
    # combine all metrics
    all_metrics = {**h_metrics, **c_metrics, **cluster_metrics, 'num_tasks': num_tasks}
    
    # print results
    print_evaluation_summary(all_metrics)
    
    # return results
    results = {'metrics': all_metrics, 'hierarchical_failed_ids': failed_ids_hierarchical, 'cem_failed_ids': failed_ids_cem}
    
    if save_dir:
        save_evaluation_results(results, save_dir)
    
    return results


def print_evaluation_summary(metrics: Dict[str, Any]):
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    # overall success rates
    print(f"\n SUCCESS RATES")
    print(f"  Hierarchical: {metrics['success_rate_hierarchical']:.2%} ({metrics['success_count_hierarchical']}/{metrics['num_tasks']})")
    if 'success_rate_cem' in metrics:
        print(f"  CEM:          {metrics['success_rate_cem']:.2%} ({metrics['success_count_cem']}/{metrics['num_tasks']})")
    
    # goal cluster reaching (hierarchical only)
    print(f"\n GOAL CLUSTER REACHING (Hierarchical)")
    print(f"  Reach Rate:   {metrics['goal_cluster_reach_rate']:.2%} ({metrics['goal_cluster_reached_count']}/{metrics['num_tasks']})")
    
    if metrics['avg_steps_to_goal_cluster'] is not None:
        print(f"  Steps to Goal Cluster: {metrics['avg_steps_to_goal_cluster']:.1f} ± {metrics['std_steps_to_goal_cluster']:.1f}")
    else:
        print(f"  Steps to Goal Cluster: N/A (no tasks reached goal cluster)")
    
    # steps for successful plans
    print(f"\n STEPS TO GOAL (Successful Plans)")
    if metrics['avg_steps_success_hierarchical'] is not None:
        print(f"  Hierarchical: {metrics['avg_steps_success_hierarchical']:.1f} ± {metrics['std_steps_success_hierarchical']:.1f}")
    else:
        print(f"  Hierarchical: N/A (no successful plans)")
    
    if 'success_rate_cem' not in metrics:
        pass
    elif metrics['avg_steps_success_cem'] is not None:
        print(f"  CEM:          {metrics['avg_steps_success_cem']:.1f} ± {metrics['std_steps_success_cem']:.1f}")
    else:
        print(f"  CEM:          N/A (no successful plans)")
    
    # failed plans statistics
    print(f"\n FAILED PLANS")
    if metrics['avg_distance_failed_hierarchical'] is not None:
        print(f"  Hierarchical: {metrics['failure_count_hierarchical']} failures, avg final distance: {metrics['avg_distance_failed_hierarchical']:.4f}")
    else:
        print(f"  Hierarchical: {metrics['failure_count_hierarchical']} failures (no failures - 100% success!)")
    
    if 'success_rate_cem' not in metrics:
        pass
    elif metrics['avg_distance_failed_cem'] is not None:
        print(f"  CEM:          {metrics['failure_count_cem']} failures, avg final distance: {metrics['avg_distance_failed_cem']:.4f}")
    else:
        print(f"  CEM:          {metrics['failure_count_cem']} failures (no failures - 100% success!)")
    
    print(f"\n{'='*80}\n")


class NumpyEncoder(json.JSONEncoder):
    """custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
   

def save_evaluation_results(results: Dict[str, Any], save_dir: str):
    """save evaluation results to JSON"""
    results_dir = os.path.join(save_dir, "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    json_results = {
        'metrics': results['metrics'],
        'hierarchical_failed_ids': results['hierarchical_failed_ids'],
        'cem_failed_ids': results['cem_failed_ids'],
    }
    
    json_path = os.path.join(results_dir, 'evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"Results saved to: {json_path}")
