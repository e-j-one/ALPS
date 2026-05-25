import numpy as np
import jax
from tqdm import tqdm
import os
import wandb

from params import TrainArgs
from utils import Buffer, sample_prior_batch
from .model import Prior


def train_gcbc_prior(prior: Prior, buffer: Buffer, args: TrainArgs, ckpt_dir: str, key: jax.random.PRNGKey, training_steps_override: int = None):
    """train goal-conditioned BC prior with variable horizon hindsight relabeling"""
    # get training hyperparameters from args
    steps = training_steps_override if training_steps_override is not None else args.prior_training_steps
    batch_size = args.batch_size
    min_horizon = args.prior_min_horizon
    max_horizon = args.prior_max_horizon

    print(f"\nTraining GCBC prior with Variable Horizon [{min_horizon}, {max_horizon}] for {steps} steps")

    # get all data in raw observation space + eigenspace
    print("Preparing episode data...")
    all_observations = buffer.get_all_observations()
    all_z = np.array(prior.processor.observations_to_eigenspace(all_observations))
    all_actions = buffer.actions[:buffer.current_size]

    # compute normalization stats from pre-computed data
    prior.compute_stats(all_observations, all_z)

    # valid episodes for sampling
    valid_mask = buffer._ep_lens_np > max_horizon
    valid_ep_starts = buffer._ep_starts_np[valid_mask]
    valid_ep_lens = buffer._ep_lens_np[valid_mask]

    if len(valid_ep_starts) == 0:
        raise ValueError(f"No episodes longer than max_horizon {max_horizon}. Longest episode: {np.max(buffer._ep_lens_np)}")
    
    rng = np.random.default_rng(int(jax.random.randint(key, (1,), 0, 2**31 - 1)[0]))

    # training loop
    for step in tqdm(range(steps), desc="Training GCBC Prior"):
        # sample batch with variable horizon hindsight relabeling
        current_obs, current_z, future_z, expert_action = sample_prior_batch(
            rng, all_observations, all_z, all_actions,
            valid_ep_starts, valid_ep_lens, batch_size, min_horizon, max_horizon
        )

        loss = prior.update(current_obs, current_z, future_z, expert_action)

        # logging
        if step % 5000 == 0 and not args.debug:
            wandb.log({
                'prior_step': step,
                'prior_loss': float(loss),
            })

    print("\nGCBC prior training completed!")

    # save checkpoint
    save_path = os.path.join(ckpt_dir, 'prior_model.pkl')
    prior.checkpoint(save_path)
    print(f"Prior saved to: {save_path}")

    return prior
