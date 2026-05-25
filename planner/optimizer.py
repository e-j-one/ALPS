import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from functools import partial

from .config import PlannerConfig
from .cost_function import compute_rollout_costs
from dynamics import Dynamics
from prior import Prior
from allo import ALLO, ALLOProcessor


def get_colored_noise(rng_key: jax.random.PRNGKey, shape: tuple, beta=0.8):
    """generate colored noise with temporal correlation"""
    noise = jax.random.normal(rng_key, shape)
    
    def smooth_step(prev, curr):
        val = beta * prev + jnp.sqrt(1 - beta**2) * curr
        return val, val

    # scan over time
    def scan_over_time(noise_seq):
        init = jnp.zeros(shape[-1])
        _, smoothed = jax.lax.scan(smooth_step, init, noise_seq)
        return smoothed

    colored_noise = jax.vmap(scan_over_time)(noise)
    return colored_noise


@partial(nnx.jit, static_argnames=['horizon', 'use_scaling', 'skip_first'])
def rollout_prior_mean(dynamics: Dynamics, prior: Prior, allo: ALLO,
                        curr_obs: jnp.ndarray, curr_z: jnp.ndarray, target_z: jnp.ndarray,
                        horizon: int, use_scaling: bool, skip_first: bool,
                        action_min: float = -1.0, action_max: float = 1.0
                        ) -> jnp.ndarray:
    """unrolls the prior with the dynamics model to generate a coherent action sequence"""
    def step_fn(carry, _):
        obs, z = carry

        # get action from prior (prior takes raw obs + direction in eigenspace)
        action = prior(obs, z, target_z)
        action = jnp.clip(action, action_min, action_max)

        # predict next obs in raw space
        next_obs = dynamics.model(obs, action)

        # project to eigenspace for next_z
        next_z = allo._get_representations(next_obs[jnp.newaxis, :], use_scaling, skip_first)[0]

        return (next_obs, next_z), action

    # scan over horizon
    _, actions = jax.lax.scan(step_fn, (curr_obs, curr_z), None, length=horizon)

    return actions


@partial(nnx.jit, static_argnames=['horizon', 'iterations', 'samples', 'num_elites', 'use_scaling', 'skip_first'])
def _cem_core(
    model: nnx.Module, allo: ALLO,
    current_obs: jax.Array, target_z: jax.Array,
    mean_trajectory: jax.Array, rng_key: jax.Array,
    horizon: int, iterations: int, samples: int, num_elites: int,
    sigma: float, beta: float, momentum: float,
    w_dist: float, w_energy: float, action_min: jax.Array, action_max: jax.Array,
    use_scaling: bool, skip_first: bool
):
    action_dim = mean_trajectory.shape[-1]
    std_trajectory = jnp.ones((horizon, action_dim)) * sigma

    # broadcast current obs for batch processing
    batch_obs = jnp.broadcast_to(current_obs, (samples, *current_obs.shape))

    def cem_iteration(carry, _):
        curr_mean, curr_std, key = carry
        key, subkey = jax.random.split(key)
        
        # sample noise
        noise = get_colored_noise(subkey, (samples, horizon, action_dim), beta=beta)
        
        # CEM samples use current_std
        candidates = curr_mean[None, :, :] + (noise * curr_std[None, :, :])
        candidates = jnp.clip(candidates, action_min, action_max)
        
        costs = compute_rollout_costs(
            model, allo, batch_obs, target_z, candidates, 
            use_scaling, skip_first, w_dist, w_energy
        )
        
        # select elites
        elite_indices = jnp.argsort(costs)[:num_elites]
        elites = candidates[elite_indices]
        
        # update mean and std - MLE of elites
        new_mean_mle = jnp.mean(elites, axis=0)
        new_std_mle = jnp.std(elites, axis=0) + 1e-6
        
        # momentum update
        new_mean = momentum * curr_mean + (1.0 - momentum) * new_mean_mle
        new_std = momentum * curr_std + (1.0 - momentum) * new_std_mle
        
        new_mean = jnp.clip(new_mean, action_min, action_max)
        new_std = jnp.clip(new_std, 0.01, 1.0)
        
        return (new_mean, new_std, key), None

    (final_mean, _, _), _ = jax.lax.scan(cem_iteration, (mean_trajectory, std_trajectory, rng_key), None, length=iterations)
    return final_mean


def optimize_trajectory(
        dynamics: Dynamics, prior: Prior,
        processor: ALLOProcessor, current_obs: jnp.ndarray,
        current_z: jnp.ndarray, target_z: jnp.ndarray, config: PlannerConfig, horizon: int,
        rng_key: jax.random.PRNGKey) -> np.ndarray:
    """optimize trajectory using specified planner (CEM or MPPI)"""
    action_dim = config.action_dim
    action_min = jnp.array(config.action_low)
    action_max = jnp.array(config.action_high)

    # warm start using prior rollout
    if config.use_prior_warmstart:
        mean_trajectory = rollout_prior_mean(
            dynamics, prior, processor.allo, current_obs, current_z, target_z,
            horizon, processor.args.use_scaling, processor.args.skip_first_eigenvector,
            action_min, action_max
        )
    else:
        mean_trajectory = jnp.zeros((horizon, action_dim))

    mean_trajectory = jnp.clip(mean_trajectory, action_min, action_max)

    final_mean = _cem_core(
        dynamics.model, processor.allo,
        current_obs, target_z, mean_trajectory, rng_key,
        horizon=horizon, iterations=config.iterations, samples=config.samples,
        num_elites=max(1, int(config.samples * config.elite_ratio)),
        sigma=config.sigma, beta=config.noise_beta, momentum=config.momentum,
        w_dist=config.distance_weight, w_energy=config.energy_weight,
        action_min=action_min, action_max=action_max,
        use_scaling=processor.args.use_scaling,
        skip_first=processor.args.skip_first_eigenvector
    )

    # return first action in the optimized mean trajectory - MPC style
    return np.array(final_mean[0])
