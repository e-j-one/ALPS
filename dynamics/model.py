import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from typing import Tuple, Dict, Any
import pickle

from params import TrainArgs
from utils import Buffer, sample_sequence_batch
from .architectures import MLP, CNNEncoder, CNNDecoder

jax.clear_caches()


class ForwardModel(nnx.Module):
    """deterministic dynamics model for flat observations"""
    def __init__(self, input_dim: int, output_dim: int, action_dim: int, hidden_dim: int, num_layers: int, rngs: nnx.Rngs):
        self.action_dim = action_dim

        # MLP for dynamics prediction
        mlp_input_dim = input_dim + action_dim
        mlp_output_dim = output_dim

        self.dynamics_net = MLP(mlp_input_dim, mlp_output_dim, hidden_dim, num_layers, rngs)

        # normalization statistics
        self.state_mean = nnx.Variable(jnp.zeros(output_dim))
        self.state_std = nnx.Variable(jnp.ones(output_dim))
        self.delta_mean = nnx.Variable(jnp.zeros(output_dim))
        self.delta_std = nnx.Variable(jnp.ones(output_dim))

    def set_stats(self, state_mean, state_std, delta_mean, delta_std):
        """set normalization statistics"""
        self.state_mean.value = jnp.array(state_mean)
        self.state_std.value = jnp.array(state_std)
        self.delta_mean.value = jnp.array(delta_mean)
        self.delta_std.value = jnp.array(delta_std)

    def __call__(self, state, action):
        """forward pass through the model"""
        # normalize input
        norm_state = (state - self.state_mean.value) / self.state_std.value
        norm_input = jnp.concatenate([norm_state, action], axis=-1)

        # predict normalized delta
        norm_delta = self.dynamics_net(norm_input)

        # denormalize delta
        delta = norm_delta * self.delta_std.value + self.delta_mean.value

        return state + delta


class ImageForwardModel(nnx.Module):
    """dynamics model for image observations with CNN encoder/decoder"""
    def __init__(self, obs_shape: Tuple[int, ...], action_dim: int, hidden_dim: int, num_layers: int, rngs: nnx.Rngs):
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        # CNN encoder
        self.encoder = CNNEncoder(obs_shape, hidden_dim, rngs)
        self.feature_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)

        # MLP for dynamics prediction in feature space
        mlp_input_dim = hidden_dim + action_dim
        mlp_output_dim = hidden_dim

        self.dynamics_net = MLP(mlp_input_dim, mlp_output_dim, hidden_dim, num_layers, rngs)

        # CNN decoder
        self.decoder = CNNDecoder(obs_shape, hidden_dim, self.encoder.conv_output_shape, self.encoder.intermediate_shapes, rngs)

    def __call__(self, image, action):
        """forward pass: image + action -> next image"""
        # check if input is unbatched (H, W, C) vs batched (batch, H, W, C)
        unbatched = image.ndim == 3
        if unbatched:
            image = image[jnp.newaxis, ...]
            action = action[jnp.newaxis, ...]

        # encode image to features and normalize
        features = self.encoder(image)
        features = self.feature_norm(features)

        # concatenate features with action
        features_action = jnp.concatenate([features, action], axis=-1)

        # predict delta in latent space (residual dynamics)
        delta_features = self.dynamics_net(features_action)

        # residual connection: z_next = z_t + delta_z
        next_features = features + delta_features

        # decode to next image
        next_image = self.decoder(next_features)

        # remove batch dim if input was unbatched
        if unbatched:
            next_image = next_image[0]

        return next_image


def autoregressive_loss(model: ForwardModel, start_state, action_seq, target_state_seq):
    """autoregressive loss function - unrolls model for H steps"""
    def step_fn(carry, inputs):
        curr_state = carry
        action, target_state = inputs

        # predict next step
        pred_state = model(curr_state, action)

        # calculate loss
        step_loss = jnp.mean((pred_state - target_state) ** 2)

        # pass predicted state for autoregressive rollout
        return pred_state, step_loss

    # transpose to (time, batch, dim) for scan
    actions_t = jnp.transpose(action_seq, (1, 0, 2))      # (H, batch, action_dim)
    targets_t = jnp.transpose(target_state_seq, (1, 0, 2))  # (H, batch, state_dim)

    _, step_losses = jax.lax.scan(
        step_fn, start_state, (actions_t, targets_t)
    )

    total_loss = jnp.mean(step_losses)
    metrics = {
        'dynamics_loss': total_loss,
        'dynamics_mse': total_loss
    }
    return total_loss, metrics


def autoregressive_loss_image(model: ImageForwardModel, start_image, action_seq, target_image_seq):
    """autoregressive loss function for images - unrolls model for H steps"""
    def step_fn(carry, inputs):
        curr_image = carry
        action, target_image = inputs

        # predict next image
        pred_image = model(curr_image, action)

        # calculate MSE loss over pixels
        step_loss = jnp.mean((pred_image - target_image) ** 2)

        # pass predicted image for autoregressive rollout
        return pred_image, step_loss

    # transpose to (time, batch, H, W, C) for scan
    actions_t = jnp.transpose(action_seq, (1, 0, 2))          # (H, batch, action_dim)
    targets_t = jnp.transpose(target_image_seq, (1, 0, 2, 3, 4))  # (H, batch, H, W, C)

    _, step_losses = jax.lax.scan(
        step_fn, start_image, (actions_t, targets_t)
    )

    total_loss = jnp.mean(step_losses)
    metrics = {
        'dynamics_loss': total_loss,
        'dynamics_mse': total_loss
    }
    return total_loss, metrics


class Dynamics(nnx.Module):
    def __init__(self, action_dim: int, buffer: Buffer, args: TrainArgs, key: jax.random.PRNGKey):
        self.args = args
        self.buffer = buffer
        self.action_dim = action_dim
        self.horizon = args.multistep_horizon
        self.obs_type = args.obs_type

        if key is not None:
            self.rngs = nnx.Rngs(int(key[0]))
        else:
            self.rngs = nnx.Rngs(0)

        # get observation shape and dimensions
        self.obs_shape = buffer.obs_shape
        self.state_dim = self.obs_shape[0]

        # model configuration
        hidden_dim = args.dynamics_hidden_dim
        num_layers = args.dynamics_num_layers
        learning_rate = args.dynamics_step_size

        # initialize model based on observation type
        if self.obs_type == 'image':
            self.model = ImageForwardModel(
                self.obs_shape, self.action_dim,
                hidden_dim, num_layers, self.rngs
            )
            # no normalization stats needed for images
            self.state_mean = None
            self.state_std = None
            self.delta_mean = None
            self.delta_std = None
        else:
            self.model = ForwardModel(
                self.state_dim, self.state_dim, self.action_dim,
                hidden_dim, num_layers, self.rngs
            )
            # wrapper-level stats (for saving/loading)
            self.state_mean = nnx.Variable(jnp.zeros(self.state_dim))
            self.state_std = nnx.Variable(jnp.ones(self.state_dim))
            self.delta_mean = nnx.Variable(jnp.zeros(self.state_dim))
            self.delta_std = nnx.Variable(jnp.ones(self.state_dim))

        self.optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate), wrt=nnx.Param)

    def compute_stats(self, buffer: Buffer):
        """compute normalization stats for raw observation space"""
        if self.obs_type == 'image':
            # images don't need normalization stats
            return

        # check if buffer has data
        if buffer.current_size == 0:
            raise ValueError("Buffer is empty. Cannot compute statistics.")

        # get all raw observations
        all_observations = buffer.observations[:buffer.current_size]

        # input stats (raw observations)
        obs_mean = jnp.mean(all_observations, axis=0)
        obs_std = jnp.std(all_observations, axis=0) + 1e-6

        # delta stats
        delta_observations = all_observations[1:] - all_observations[:-1]

        # apply valid mask to exclude episode boundaries
        valid_mask = buffer.time_to_end[:buffer.current_size][:-1] > 0
        valid_deltas = delta_observations[valid_mask]

        # calculate delta stats
        delta_mean = jnp.mean(valid_deltas, axis=0)
        delta_std = jnp.std(valid_deltas, axis=0) + 1e-6

        # set stats in model
        self.set_statistics(obs_mean, obs_std, delta_mean, delta_std)
        print("Raw Observation Statistics Set")

    def set_statistics(self, obs_mean, obs_std, delta_mean, delta_std):
        """set normalization statistics (flat obs only)"""
        if self.obs_type == 'image':
            return

        self.state_mean.value = jnp.array(obs_mean)
        self.state_std.value = jnp.array(obs_std)
        self.delta_mean.value = jnp.array(delta_mean)
        self.delta_std.value = jnp.array(delta_std)

        # set stats in the inner model
        self.model.set_stats(self.state_mean.value, self.state_std.value, self.delta_mean.value, self.delta_std.value)

    @nnx.jit
    def _update_step_flat(self, start_states: jnp.ndarray, action_seqs: jnp.ndarray, target_state_seqs: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """update step for flat observations"""
        (loss, metrics), grads = nnx.value_and_grad(
            autoregressive_loss, has_aux=True
        )(self.model, start_states, action_seqs, target_state_seqs)

        self.optimizer.update(self.model, grads)
        return loss, metrics

    @nnx.jit
    def _update_step_image(self, start_images: jnp.ndarray, action_seqs: jnp.ndarray, target_image_seqs: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """update step for image observations"""
        (loss, metrics), grads = nnx.value_and_grad(
            autoregressive_loss_image, has_aux=True
        )(self.model, start_images, action_seqs, target_image_seqs)

        self.optimizer.update(self.model, grads)
        return loss, metrics

    def update_step(self, start_states: jnp.ndarray, action_seqs: jnp.ndarray, target_state_seqs: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """update step with autoregressive rollout - dispatches to appropriate method"""
        if self.obs_type == 'image':
            return self._update_step_image(start_states, action_seqs, target_state_seqs)
        else:
            return self._update_step_flat(start_states, action_seqs, target_state_seqs)

    def checkpoint(self, filepath: str):
        """save dynamics model checkpoint"""
        _, state = nnx.split(self.model)
        flat_state = dict(state.flat_state())
        state_dict = {'/'.join(map(str, k)): np.array(v.value) for k, v in flat_state.items()}

        save_data = {
            'state_dict': state_dict,
            'obs_type': self.obs_type,
            'config': {
                'state_dim': self.state_dim,
                'obs_shape': self.obs_shape,
                'horizon': self.horizon
            }
        }

        # only save stats for flat observations
        if self.obs_type != 'image':
            save_data['stats'] = {
                'state_mean': np.array(self.state_mean.value),
                'state_std': np.array(self.state_std.value),
                'delta_mean': np.array(self.delta_mean.value),
                'delta_std': np.array(self.delta_std.value)
            }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load_checkpoint(cls, filepath: str, action_dim: int, buffer: Buffer, args: TrainArgs):
        """load dynamics model from checkpoint"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        state_dict = data['state_dict']
        config = data['config']
        obs_type = data.get('obs_type', 'xy')  # default to 'xy' for backwards compatibility

        print(f"Dynamics model loaded successfully from: {filepath}")
        print(f"  obs_type: {obs_type}")

        # verify dimensions match
        expected_state_dim = buffer.obs_shape[0]
        if config.get('state_dim') and config['state_dim'] != expected_state_dim:
            print(f"WARNING: State dim mismatch in loaded model.")

        if config.get('horizon') and config['horizon'] != args.multistep_horizon:
            print(f"WARNING: Horizon mismatch! Args give {args.multistep_horizon}, checkpoint has {config['horizon']}")

        # create fresh instance
        wrapper = cls(action_dim, buffer, args, jax.random.PRNGKey(0))

        # restore state by directly updating values
        _, state = nnx.split(wrapper.model)

        # normalize saved keys to tuple format for matching
        def to_tuple_key(k):
            if isinstance(k, str):
                parts = k.split('/')
                return tuple(int(p) if p.isdigit() else p for p in parts)
            return k

        normalized_dict = {to_tuple_key(k): v for k, v in state_dict.items()}

        for path, var_state in state.flat_state():
            if path in normalized_dict:
                var_state.value = jnp.array(normalized_dict[path])

        # restore stats for flat observations only
        if obs_type != 'image' and 'stats' in data:
            stats = data['stats']
            wrapper.state_mean.value = jnp.array(stats['state_mean'])
            wrapper.state_std.value = jnp.array(stats['state_std'])
            wrapper.delta_mean.value = jnp.array(stats['delta_mean'])
            wrapper.delta_std.value = jnp.array(stats['delta_std'])
            # also set in inner model
            wrapper.model.set_stats(wrapper.state_mean.value, wrapper.state_std.value,
                                    wrapper.delta_mean.value, wrapper.delta_std.value)

        wrapper.optimizer = nnx.Optimizer(wrapper.model, optax.adam(args.dynamics_step_size), wrt=nnx.Param)

        return wrapper

    def test_dynamics(self, buffer: Buffer, num_test_samples: int = 1000):
        """test dynamics model predictions with autoregressive rollout"""
        print(f"\nTesting Dynamics (Horizon={self.horizon}, obs_type={self.obs_type})...")

        # prepare episode data
        valid_mask = buffer._ep_lens_np >= (self.horizon + 1)
        valid_ep_starts = buffer._ep_starts_np[valid_mask]
        valid_ep_lens = buffer._ep_lens_np[valid_mask]

        # precompute all data
        all_observations = buffer.observations[:buffer.current_size]
        all_actions = buffer.actions[:buffer.current_size]

        # sample test batch
        test_rng = np.random.default_rng(42)
        batch_start_obs, batch_actions, batch_target_obs = sample_sequence_batch(
            test_rng, all_observations, all_actions, valid_ep_starts, valid_ep_lens,
            num_test_samples, self.horizon
        )

        print(f"  Testing on {num_test_samples} sequences of length {self.horizon}")

        # run autoregressive rollout and compute MSE at each step
        @nnx.jit
        def run_test_rollout(model, start, actions, targets):
            def step_fn(current_obs, inputs):
                action, target = inputs
                pred_obs = model(current_obs, action)
                
                # compute MSE based on shape
                diff = pred_obs - target
                mse = jnp.mean(diff ** 2)
                
                return pred_obs, mse
            
            actions_t = jnp.transpose(actions, (1, 0, 2))
            
            if self.obs_type == 'image':
                targets_t = jnp.transpose(targets, (1, 0, 2, 3, 4))
            else:
                targets_t = jnp.transpose(targets, (1, 0, 2))

            _, step_errors = jax.lax.scan(step_fn, start, (actions_t, targets_t))
            return step_errors

        # run test rollout
        step_errors_jax = run_test_rollout(self.model, batch_start_obs, batch_actions, batch_target_obs)
        step_errors = np.array(step_errors_jax).tolist()
        avg_mse = np.mean(step_errors)

        print(f"\n  Prediction Errors (Autoregressive):")
        print(f"  {'Step':<6} {'MSE':<12}")
        print(f"  {'-'*6} {'-'*12}")
        for t, err in enumerate(step_errors):
            print(f"  {t+1:<6} {err:<12.6f}")
        print(f"\n  Average MSE: {avg_mse:.6f}")

        return {'step_errors': step_errors, 'avg_mse': float(avg_mse)}
