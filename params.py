from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainArgs:
    """training parameters"""
    # === ENVIRONMENT ===
    env_type: str = 'OGBenchEnv'                    # 'RoomEnv' or 'OGBenchEnv'

    # RoomEnv specific
    rooms_env_name: str = 'rooms'                   # 'rooms', 'hallway', 'spiral'
    room_size: int = 100
    max_episode_steps: int = 30
    obs_type: str = 'xy'                            # 'xy' or 'image'

    # OGBenchEnv specific
    ogbench_env_type: str = 'AntMaze'               # 'PointMaze', 'AntMaze', 'HumanoidMaze'
    ogbench_task_name: str = 'antmaze-medium-navigate-v0'  # 'pointmaze-teleport-navigate-v0', 'antmaze-giant-navigate-v0'. 'antmaze-large-navigate-v0'
    
    # === DATA COLLECTION ===
    load_offline_dataset: bool = True
    buffer_size: int = 500000

    # === MODEL ARCHITECTURE ===
    
    # ALLO
    num_eigenvectors: int = 32
    allo_hidden_dim: int = 256
    use_discounted_sampling: bool = True
    sampling_discount: float = 0.6
    use_scaling: bool = True
    skip_first_eigenvector: bool = True
    
    # dynamics
    dynamics_hidden_dim: int = 512
    dynamics_num_layers: int = 4
    multistep_horizon: int = 10

    # prior
    prior_hidden_dim: int = 512
    prior_num_layers: int = 4
    prior_min_horizon: int = 1
    prior_max_horizon: int = 50
    
    # === TRAINING HYPERPARAMETERS ===
    # training steps
    allo_training_steps: int = 1000000
    dynamics_training_steps: int = 1000000
    prior_training_steps: int = 1000000

    # checkpoint fractions (0.8, 0.9, 1.0])
    checkpoint_fractions: tuple = (1.0,)
    
    # learning rates
    allo_step_size: float = 1e-4
    dynamics_step_size: float = 3e-4
    prior_step_size: float = 3e-4
    
    # ALLO dual parameters
    duals_initial_val: float = -1.0
    barrier_initial_val: float = 0.5
    min_duals: float = -100.0
    max_duals: float = 100.0
    min_barrier_coefs: float = 0.0
    max_barrier_coefs: float = 0.5
    step_size_duals: float = 1.0

    # batch size
    batch_size: int = 1024
    
    # === BASIC CONFIG ===
    debug: bool = False
    seed: int = 14
    save_dir: str = './results'
    num_tasks: int = 100


@dataclass
class EvalArgs:
    """evaluation parameters"""
    # test models
    show_eigenvalues: bool = False
    test_dynamics: bool = False
    
    # clustering
    num_clusters: int = 16

    # cluster graph
    top_p: float = 0.95

    # planning
    start_position: Optional[list] = None
    goal_position: Optional[list] = None

    # planner parameters
    use_planner: bool = True
    use_prior_warmstart: bool = True
    horizon: int = 20
    iterations: int = 5
    samples: int = 500
    momentum: float = 0.3
    sigma: float = 0.5
    noise_beta: float = 0.9
    elite_ratio: float = 0.15
    
    # cost weights
    distance_weight: float = 1.0
    energy_weight: float = 0.01
    
    # evaluation
    get_success_rate: bool = False
    eval_checkpoint: int = 100              # checkpoint selection for evaluation

    # === RoomEnv specific ===
    tasks_dir: Optional[str] = "./tasks"
    goal_tolerance: float = 0.030
    
    # visualization
    show_graph: bool = False
    render : bool = False


@dataclass
class Args(TrainArgs, EvalArgs):
    """experiment args"""
    train: bool = False
    test: bool = False
    use_wandb: bool = False
    project_name: str = 'ALPS'