import os
from typing import Dict
import numpy as np
import jax
import wandb
from tqdm import tqdm

from params import TrainArgs
from utils import Buffer, sample_sequence_batch
from .model import Dynamics


def train_dynamics(dynamics: Dynamics, buffer: Buffer, args: TrainArgs, ckpt_dir: str, key: jax.random.PRNGKey, checkpoint_dirs: Dict[int, str] = None):
    """dynamics training function with autoregressive multi-step rollouts"""
    print(f"\nTraining Dynamics Model (Horizon={dynamics.horizon})")

    # determine checkpoint steps
    if checkpoint_dirs is not None:
        checkpoint_steps = sorted(checkpoint_dirs.keys())
        print(f"  Checkpoints at steps: {checkpoint_steps}")
    else:
        checkpoint_steps = []

    # prepare episode metadata
    print("Preparing episode data...")

    # filter episodes long enough for multi-step training
    valid_mask = buffer._ep_lens_np >= (dynamics.horizon + 1)
    valid_ep_starts = buffer._ep_starts_np[valid_mask]
    valid_ep_lens = buffer._ep_lens_np[valid_mask]

    # get all data from buffer
    all_observations = buffer.observations[:buffer.current_size]
    all_actions = buffer.actions[:buffer.current_size]

    # compute normalization statistics
    dynamics.compute_stats(buffer)
    
    rng = np.random.default_rng(int(jax.random.randint(key, (1,), 0, 2**31 - 1)[0]))

    # training loop
    for step in tqdm(range(args.dynamics_training_steps), desc="Training Dynamics"):
        # sample batch of sequences
        batch_start_states, batch_actions, batch_target_states = sample_sequence_batch(
            rng, all_observations, all_actions, valid_ep_starts,
            valid_ep_lens, args.batch_size, dynamics.horizon
        )

        # update with autoregressive loss
        loss, metrics = dynamics.update_step(batch_start_states, batch_actions, batch_target_states)

        if step % 5000 == 0 and not args.debug:
            wandb.log({
                'dyn_step': step,
                'dynamics_loss': float(metrics['dynamics_loss']),
                'dynamics_mse': float(metrics['dynamics_mse'])
            })

        # save checkpoint if at checkpoint step
        current_step = step + 1
        if current_step in checkpoint_steps:
            ckpt_path = os.path.join(checkpoint_dirs[current_step], 'dynamics_model.pkl')
            dynamics.checkpoint(ckpt_path)
            print(f"\n  Dynamics checkpoint saved at step {current_step} to {ckpt_path}")

    print("Dynamics training completed!")

    # save final checkpoint
    if checkpoint_dirs is None:
        dynamics_save_path = os.path.join(ckpt_dir, 'dynamics_model.pkl')
        dynamics.checkpoint(dynamics_save_path)
        print(f"Dynamics model saved to {dynamics_save_path}")

    return dynamics
