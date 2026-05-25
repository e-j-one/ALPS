import jax
import jax.numpy as jnp
import flax.nnx as nnx

from allo import ALLO


def compute_rollout_costs(model: nnx.Module, allo: ALLO, start_obs: jnp.ndarray,
        target_z: jnp.ndarray, action_sequences: jnp.ndarray,
        use_scaling: bool, skip_first: bool,
        dist_weight: float, energy_weight: float
    ) -> jnp.ndarray:
    """rollout cost function using raw obs dynamics with eigenspace cost computation"""
    # transpose for scan: (horizon, samples, action_dim)
    action_sequences_T = jnp.transpose(action_sequences, (1, 0, 2))

    initial_carry = start_obs

    def step_fn(curr_obs, action):
        # predict next obs in raw space
        next_obs = model(curr_obs, action)

        # project to eigenspace for cost computation
        next_z = jax.vmap(lambda s: allo._get_representations(s[jnp.newaxis, :], use_scaling, skip_first)[0])(next_obs)

        # distance cost to target in eigenspace
        diff = next_z - target_z[None, :]
        dist_cost = jnp.sum(jnp.square(diff), axis=1)

        # energy cost for action taken
        energy_cost = jnp.sum(jnp.square(action), axis=1)
        
        # weighted sum of costs
        step_cost = (dist_weight * dist_cost) + (energy_weight * energy_cost)

        return next_obs, step_cost

    # scan over time steps
    _, costs_per_step = jax.lax.scan(step_fn, initial_carry, action_sequences_T)
    
    # compute total cost per trajectory
    total_trajectory_cost = jnp.sum(costs_per_step, axis=0)
    
    return total_trajectory_cost
