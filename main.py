import os
import random
from functools import partial
import tyro
import wandb
import numpy as np
import jax
from typing import Union
import gymnasium as gym
import ogbench

from params import Args
from envs import RoomEnv, setup_environment, is_sharded_ogbench_dataset, list_ogbench_shards, load_ogbench_shard
from utils import Buffer, DiscountedReplayBuffer, EnvironmentHelper, setup_logging
from allo import ALLO, ALLOProcessor, train_allo
from dynamics import Dynamics, train_dynamics
from prior import Prior, train_gcbc_prior
from clustering import SpectralClustering
from graph import ClusterGraph
from planner import Planner
from evaluation import evaluate_planners


def get_buffer(env: Union[RoomEnv, gym.Env], args: Args, obs_shape: tuple, action_dim: int, key: jax.random.PRNGKey) -> Buffer:
    """load/generate data"""
    if args.env_type == "OGBenchEnv" and args.load_offline_dataset and is_sharded_ogbench_dataset(args.ogbench_task_name):
        # large sharded dataset: hold one shard in memory and rotate during training (as in horizon-reduction)
        shard_paths = list_ogbench_shards(os.path.join('./data', args.ogbench_task_name))
        print(f"\nLoading sharded dataset: {len(shard_paths)} shards, replacing every {args.dataset_replace_interval} steps")
        loader = partial(load_ogbench_shard, dataset_name=args.ogbench_task_name)

        buffer = DiscountedReplayBuffer(args, 0, obs_shape, action_dim, key, allocate=False)
        buffer.load_offline_dataset(loader(shard_paths[0]))
        buffer.enable_shard_rotation(shard_paths, args.dataset_replace_interval, loader)
        print(f"Loaded shard 1/{len(shard_paths)} with {buffer.num_episodes} episodes")

        return buffer
    elif args.env_type == "OGBenchEnv" and args.load_offline_dataset:
        print("\nLoading offline dataset...")
        _, train_dataset, _ = ogbench.make_env_and_datasets(args.ogbench_task_name, dataset_dir='./data', compact_dataset=True)

        buffer = DiscountedReplayBuffer(args, 0, obs_shape, action_dim, key, allocate=False)
        buffer.load_offline_dataset(train_dataset)
        del train_dataset
        print(f"Loaded dataset with {buffer.num_episodes} episodes")

        return buffer
    else:
        print("Generating data...")
        buffer = DiscountedReplayBuffer(args, args.buffer_size, obs_shape, action_dim, key)
        buffer.generate_continuous_buffer(env)
        print(f"Generated buffer of {buffer.num_episodes} episodes")

        return buffer


def train(buffer: Buffer, args: Args, model_dir: str, obs_shape: tuple, action_dim: int, key: jax.random.PRNGKey, checkpoint_dirs: dict):
    """ALLO, dynamics, and prior training function"""
    print("\n" + "="*50)
    allo_key, dynamics_key, prior_key = jax.random.split(key, 3)

    # initialize models
    allo = ALLO(obs_shape[0], obs_shape, args, allo_key)
    dynamics = Dynamics(action_dim, buffer, args, dynamics_key)

    # create step -> dir mappings from pct -> dir mapping
    checkpoint_fractions = args.checkpoint_fractions
    allo_ckpt_dirs = {int(frac * args.allo_training_steps): checkpoint_dirs[int(frac * 100)] for frac in checkpoint_fractions}
    dynamics_ckpt_dirs = {int(frac * args.dynamics_training_steps): checkpoint_dirs[int(frac * 100)] for frac in checkpoint_fractions}

    # train allo
    buffer.reset_shard()
    allo = train_allo(allo, buffer, args, model_dir, allo_key, checkpoint_dirs=allo_ckpt_dirs)

    print("\n" + "="*50)

    # train dynamics
    buffer.reset_shard()
    dynamics = train_dynamics(dynamics, buffer, args, model_dir, dynamics_key, checkpoint_dirs=dynamics_ckpt_dirs)

    print("\n" + "="*50)

    # train prior for each checkpoint
    for frac in checkpoint_fractions:
        pct = int(frac * 100)
        ckpt_subdir = checkpoint_dirs[pct]
        prior_steps = int(frac * args.prior_training_steps)

        print(f"\n{'='*50}")
        print(f"Training prior for checkpoint_{pct} ({prior_steps} steps)")

        allo_load_path = os.path.join(ckpt_subdir, 'allo_model.pkl')
        print(f"  Loading ALLO from {allo_load_path}")
        allo_for_prior = ALLO.load_checkpoint(allo_load_path, obs_shape[0], obs_shape, args)
        processor_for_prior = ALLOProcessor(allo_for_prior, args)

        prior_key, subkey = jax.random.split(prior_key)
        buffer.reset_shard()
        fresh_prior = Prior(buffer, processor_for_prior, args, subkey)
        fresh_prior = train_gcbc_prior(
            fresh_prior, buffer, args, ckpt_subdir, subkey,
            training_steps_override=prior_steps
        )

        print(f"  prior for checkpoint_{pct} completed!")

    print("\n" + "="*50)


def load_models( buffer: Buffer, args: Args, obs_shape: tuple, action_dim: int, checkpoint_dirs: dict):
    """load trained models from checkpoint"""
    pct = args.eval_checkpoint
    if pct not in checkpoint_dirs:
        raise ValueError(f"Checkpoint {pct}% not found. Available: {sorted(checkpoint_dirs.keys())}")
    load_dir = checkpoint_dirs[pct]
    print(f"Loading models from: {load_dir}")

    allo = ALLO.load_checkpoint(os.path.join(load_dir, 'allo_model.pkl'), obs_shape[0], obs_shape, args)
    processor = ALLOProcessor(allo, args)
    dynamics = Dynamics.load_checkpoint(os.path.join(load_dir, 'dynamics_model.pkl'), action_dim, buffer, args)
    prior = Prior.load_checkpoint(os.path.join(load_dir, 'prior_model.pkl'), buffer, processor, args)
    print("Models loaded!")

    if args.show_eigenvalues:
        eigenvalue_dir = os.path.join(load_dir, "eigenvalues")
        os.makedirs(eigenvalue_dir, exist_ok=True)
        processor.plot_eigenvalues(save_dir=eigenvalue_dir)

    if args.test_dynamics:
        print("\nTesting dynamics model predictions...")
        dynamics.test_dynamics(buffer, num_test_samples=100)

    return processor, dynamics, prior


def spectral_clustering(env: Union[RoomEnv, gym.Env], buffer: Buffer, processor: ALLOProcessor, eval_dir: str, args: Args, key: jax.random.PRNGKey) -> SpectralClustering:
    """spectral clustering with ALLO representations"""
    eval_obs = buffer.get_all_observations()
    allo_clustering = SpectralClustering(env, processor, args, eval_dir, key)
    allo_clustering.perform_spectral_clustering(eval_obs, args.num_clusters)

    return allo_clustering


def generate_graph(env: Union[RoomEnv, gym.Env], buffer: Buffer, clustering: SpectralClustering, eval_dir: str, args: Args) -> ClusterGraph:
    """get cluster connectivity graph"""
    graph = ClusterGraph(env, clustering, args, eval_dir)
    graph.create_cluster_graph(buffer, args.top_p)

    return graph


def planning(env: Union[RoomEnv, gym.Env], env_helper: EnvironmentHelper, processor: ALLOProcessor, dynamics: Dynamics, prior: Prior,
              clustering: SpectralClustering, cluster_graph: ClusterGraph, eval_dir: str, args: Args, key: jax.random.PRNGKey) -> Planner:
    """planning using hierarchical and cem planners"""
    planner = Planner(env, env_helper, processor, dynamics, prior, clustering, cluster_graph, args, eval_dir, key, save_video=True)

    hierarchical_planner = planner.get_hierarchical_planner()
    cem_planner = planner.get_cem_planner()

    start_position = args.start_position
    goal_position = args.goal_position

    print(f"\nRunning hierarchical planning...")
    if args.env_type == 'OGBenchEnv':
        hierarchical_planner.plan(task_id=1, record_video=args.render)
    else:
        hierarchical_planner.plan(start_position, goal_position, record_video=args.render)

    if args.eval_cem_planner:
        print(f"\nRunning CEM planning...")
        if args.env_type == 'OGBenchEnv':
            cem_planner.plan(task_id=1, record_video=args.render)
        else:
            cem_planner.plan(start_position, goal_position, record_video=args.render)

    return planner


def main(args: Args) -> str:
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng_key = jax.random.PRNGKey(args.seed)
    buffer_key, train_key, clustering_key, planner_key = jax.random.split(rng_key, 4)

    # setup environment
    env = setup_environment(args, render_mode='rgb_array' if args.render else None)
    env_helper = EnvironmentHelper(env, args)
    obs_shape = env_helper.state_shape
    action_dim = env_helper.action_dim

    # get replay buffer
    buffer = get_buffer(env, args, obs_shape, action_dim, buffer_key)

    # setup logging
    model_dir, eval_dir, checkpoint_dirs = setup_logging(args)

    # train models
    if args.train:
        print("\nTraining ALLO, dynamics, and prior...")
        train(buffer, args, model_dir, obs_shape, action_dim, train_key, checkpoint_dirs)
        print("\nTraining completed!")

    # load and evaluate models
    if args.test:
        buffer.reset_shard()
        print("\n" + "="*50)
        processor, dynamics, prior = load_models(buffer, args, obs_shape, action_dim, checkpoint_dirs)

        print("\n" + "="*50)
        print("\nPerforming spectral clustering...")
        clustering = spectral_clustering(env, buffer, processor, eval_dir, args, clustering_key)

        print("\n" + "="*50)
        print("\nGenerating cluster graph...")
        cluster_graph = generate_graph(env, buffer, clustering, eval_dir, args)

        print("\n" + "="*50)
        print("\nPlanning in eigenspace...")
        planner = planning(env, env_helper, processor, dynamics, prior, clustering, cluster_graph, eval_dir, args, planner_key)

        if args.get_success_rate:
            print("\n" + "="*50)
            print("\nEvaluating agent success rate...")
            evaluate_planners(planner, args, eval_dir)

        print("\nDONE!\n")

    wandb.finish()
    return eval_dir


if __name__ == "__main__":
    wandb.require("legacy-service")
    print("JAX devices:", jax.devices())
    print("JAX default device:", jax.devices()[0])

    args = tyro.cli(Args)

    print("="*70)
    print(f"Configuration:")
    print(f"  Environment type: {args.env_type}")
    print(f"  Seed: {args.seed}")
    print("="*70)

    main(args)
