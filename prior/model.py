import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import pickle
import numpy as np
from typing import Tuple

from utils import Buffer, compiled_flops, sds
from allo import ALLOProcessor
from params import TrainArgs


class MLP(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, rngs: nnx.Rngs):
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nnx.Linear(current_dim, hidden_dim, rngs=rngs))
            layers.append(nnx.relu)
            current_dim = hidden_dim
        layers.append(nnx.Linear(current_dim, output_dim, rngs=rngs))
        self.net = nnx.Sequential(*layers)
    
    def __call__(self, x):
        return self.net(x)


class CNNEncoder(nnx.Module):
    """CNN encoder for image observations"""
    def __init__(self, input_shape: Tuple[int, ...], hidden_dim: int, rngs: nnx.Rngs):
        self.input_shape = input_shape

        # conv layers
        self.conv1 = nnx.Conv(input_shape[-1], 16, kernel_size=(4, 4), strides=2, padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(4, 4), strides=2, padding='SAME', rngs=rngs)
        self.conv3 = nnx.Conv(32, 32, kernel_size=(4, 4), strides=2, padding='SAME', rngs=rngs)
        self.conv4 = nnx.Conv(32, 32, kernel_size=(4, 4), strides=2, padding='SAME', rngs=rngs)

        # compute output shape via dummy forward pass
        dummy_input = jnp.zeros((1, *input_shape))
        out = self.conv4(self.conv3(self.conv2(self.conv1(dummy_input))))
        self.conv_output_shape = out.shape[1:]
        conv_output_size = int(jnp.prod(jnp.array(self.conv_output_shape)))
        self.flatten_linear = nnx.Linear(conv_output_size, hidden_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = nnx.relu(self.conv4(x))
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.flatten_linear(x)
        return x


class Prior(nnx.Module):
    def __init__(self, buffer: Buffer, processor: ALLOProcessor, args: TrainArgs, key: jax.random.PRNGKey):
        self.args = args
        self.buffer = buffer
        self.processor = processor
        self.obs_type = args.obs_type

        if key is not None:
            self.rngs = nnx.Rngs(int(key[0]))
        else:
            self.rngs = nnx.Rngs(0)

        # determine dimensions
        if args.skip_first_eigenvector:
            self.z_dim = args.num_eigenvectors
        else:
            self.z_dim = args.num_eigenvectors + 1

        # get observation shape and dimensions
        self.obs_shape = buffer.obs_shape
        self.state_dim = self.obs_shape[0]
        self.action_dim = buffer.action_dim

        # model configuration from args
        hidden_dim = args.prior_hidden_dim
        num_layers = args.prior_num_layers
        learning_rate = args.prior_step_size

        # initialize based on observation type
        if self.obs_type == 'image':
            # CNN encoder for images
            self.encoder = CNNEncoder(self.obs_shape, hidden_dim, self.rngs)
            # input: [encoded_features, z_t, z_{t+k}]
            input_dim = hidden_dim + 2 * self.z_dim
            # no state normalization for images
            self.state_mean = None
            self.state_std = None
        else:
            self.encoder = None
            # input: [current_state, z_t, z_{t+k}]
            input_dim = self.state_dim + 2 * self.z_dim
            # normalization parameters for flat observations
            self.state_mean = nnx.Variable(jnp.zeros(self.state_dim))
            self.state_std = nnx.Variable(jnp.ones(self.state_dim))
        
        raw_mlp = MLP(input_dim, self.action_dim, hidden_dim, num_layers, self.rngs)
        if self.obs_type == 'image':
            self.net = nnx.Sequential(nnx.LayerNorm(input_dim, rngs=self.rngs), raw_mlp)
        else:
            self.net = raw_mlp
        
        self.optimizer = nnx.Optimizer(self.net, optax.adam(learning_rate), wrt=nnx.Param)

        # separate optimizer for encoder
        if self.obs_type == 'image':
            self.encoder_optimizer = nnx.Optimizer(self.encoder, optax.adam(learning_rate), wrt=nnx.Param)
        else:
            self.encoder_optimizer = None

        # z normalization parameters
        self.z_mean = nnx.Variable(jnp.zeros(self.z_dim))
        self.z_std = nnx.Variable(jnp.ones(self.z_dim))

    def compute_stats(self, all_observations: jnp.ndarray, all_z: jnp.ndarray):
        """compute normalization statistics"""
        if self.obs_type == 'image':
            print("Image observations: no state normalization needed for prior")
        else:
            # raw observation stats for flat observations
            self.state_mean[...] = jnp.mean(all_observations, axis=0)
            self.state_std[...] = jnp.std(all_observations, axis=0) + 1e-6

        # eigenspace stats
        self.z_mean[...] = jnp.mean(all_z, axis=0)
        self.z_std[...] = jnp.std(all_z, axis=0) + 1e-6

        print("Prior Stats Set")

    def _normalize_z(self, current_z, target_z):
        """normalize z representations"""
        norm_current_z = (current_z - self.z_mean[...]) / self.z_std[...]
        norm_target_z = (target_z - self.z_mean[...]) / self.z_std[...]
        return norm_current_z, norm_target_z

    def _normalize_inputs_flat(self, current_obs, current_z, target_z):
        """normalize inputs for flat observations"""
        norm_obs = (current_obs - self.state_mean[...]) / self.state_std[...]
        norm_current_z, norm_target_z = self._normalize_z(current_z, target_z)
        return norm_obs, norm_current_z, norm_target_z

    def _encode_and_normalize_image(self, current_obs, current_z, target_z):
        """encode image and normalize z for image observations"""
        encoded_obs = self.encoder(current_obs)
        norm_current_z, norm_target_z = self._normalize_z(current_z, target_z)
        return encoded_obs, norm_current_z, norm_target_z

    def __call__(self, current_obs, current_z, target_z):
        """predict action given current observation, current z, and target z"""
        if self.obs_type == 'image':
            # check if input is unbatched (H, W, C) vs batched (batch, H, W, C)
            unbatched = current_obs.ndim == 3
            if unbatched:
                current_obs = current_obs[jnp.newaxis, ...]
                current_z = current_z[jnp.newaxis, ...]
                target_z = target_z[jnp.newaxis, ...]

            encoded_obs, norm_current_z, norm_target_z = self._encode_and_normalize_image(current_obs, current_z, target_z)
            inputs = jnp.concatenate([encoded_obs, norm_current_z, norm_target_z], axis=-1)
            
            output = self.net(inputs)

            if unbatched:
                output = output[0]
            return output
        else:
            norm_obs, norm_current_z, norm_target_z = self._normalize_inputs_flat(current_obs, current_z, target_z)
            inputs = jnp.concatenate([norm_obs, norm_current_z, norm_target_z], axis=-1)
            return self.net(inputs)

    @nnx.jit
    def _update_flat(self, current_obs, current_z, future_z, expert_action):
        """training step for flat observations"""
        def loss_fn(net):
            norm_obs, norm_current_z, norm_future_z = self._normalize_inputs_flat(current_obs, current_z, future_z)
            inputs = jnp.concatenate([norm_obs, norm_current_z, norm_future_z], axis=-1)
            pred_action = net(inputs)
            return jnp.mean((pred_action - expert_action) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(self.net)
        self.optimizer.update(self.net, grads)
        return loss

    @nnx.jit
    def _update_image(self, current_obs, current_z, future_z, expert_action):
        """training step for image observations"""
        def loss_fn(net, encoder):
            encoded_obs = encoder(current_obs)
            norm_current_z, norm_future_z = self._normalize_z(current_z, future_z)
            inputs = jnp.concatenate([encoded_obs, norm_current_z, norm_future_z], axis=-1)
            pred_action = net(inputs)
            return jnp.mean((pred_action - expert_action) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn, argnums=(0, 1))(self.net, self.encoder)
        self.optimizer.update(self.net, grads[0])
        self.encoder_optimizer.update(self.encoder, grads[1])
        return loss

    def update(self, current_obs, current_z, future_z, expert_action):
        """training step"""
        if self.obs_type == 'image':
            return self._update_image(current_obs, current_z, future_z, expert_action)
        else:
            return self._update_flat(current_obs, current_z, future_z, expert_action)

    def step_flops(self, batch_size: int) -> float:
        """FLOPs of one update step (lowered from shapes only)"""
        obs = sds((batch_size,) + tuple(self.obs_shape))
        z = sds((batch_size, self.z_dim))
        action = sds((batch_size, self.action_dim))
        update_fn = Prior._update_image if self.obs_type == 'image' else Prior._update_flat
        return compiled_flops(update_fn, self, obs, z, z, action)

    def checkpoint(self, filepath: str):
        """save prior checkpoint"""
        # get state from net
        _, net_state = nnx.split(self.net)
        net_flat = dict(nnx.to_flat_state(net_state))
        # convert tuple keys to string paths for compatibility with replace_by_pure_dict
        net_state_dict = {'/'.join(map(str, k)): np.array(v[...]) for k, v in net_flat.items()}

        save_data = {
            'net_state_dict': net_state_dict,
            'obs_type': self.obs_type,
            'stats': {
                'z_mean': np.array(self.z_mean[...]),
                'z_std': np.array(self.z_std[...]),
            },
            'config': {
                'state_dim': self.state_dim,
                'obs_shape': self.obs_shape,
                'z_dim': self.z_dim,
                'action_dim': self.action_dim
            }
        }

        # save encoder state for images
        if self.obs_type == 'image':
            _, encoder_state = nnx.split(self.encoder)
            encoder_flat = dict(nnx.to_flat_state(encoder_state))
            save_data['encoder_state_dict'] = {'/'.join(map(str, k)): np.array(v[...]) for k, v in encoder_flat.items()}
        else:
            save_data['stats']['state_mean'] = np.array(self.state_mean[...])
            save_data['stats']['state_std'] = np.array(self.state_std[...])

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load_checkpoint(cls, filepath: str, buffer: Buffer, processor: ALLOProcessor, args: TrainArgs):
        """load prior from checkpoint using nnx.update for proper state restoration"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        net_state_dict = data['net_state_dict']
        stats = data['stats']
        config = data['config']
        obs_type = data.get('obs_type', 'xy')

        print(f"Prior loaded from: {filepath}")

        # verify dimensions match
        expected_state_dim = buffer.obs_shape[0]
        if config.get('state_dim') and config['state_dim'] != expected_state_dim:
            print(f"WARNING: state_dim mismatch - checkpoint has {config['state_dim']}, buffer has {expected_state_dim}")

        # create fresh instance
        wrapper = cls(buffer, processor, args, jax.random.PRNGKey(0))

        # helper to normalize saved keys to tuple format for matching
        def to_tuple_key(k):
            if isinstance(k, str):
                parts = k.split('/')
                return tuple(int(p) if p.isdigit() else p for p in parts)
            return k

        # restore net state by directly updating values
        _, net_state = nnx.split(wrapper.net)
        normalized_net_dict = {to_tuple_key(k): v for k, v in net_state_dict.items()}
        for path, var_state in nnx.to_flat_state(net_state):
            if path in normalized_net_dict:
                var_state[...] = jnp.array(normalized_net_dict[path])

        # restore encoder state for images
        if obs_type == 'image' and 'encoder_state_dict' in data:
            encoder_state_dict = data['encoder_state_dict']
            _, encoder_state = nnx.split(wrapper.encoder)
            normalized_encoder_dict = {to_tuple_key(k): v for k, v in encoder_state_dict.items()}
            for path, var_state in nnx.to_flat_state(encoder_state):
                if path in normalized_encoder_dict:
                    var_state[...] = jnp.array(normalized_encoder_dict[path])

        # restore stats
        wrapper.z_mean[...] = jnp.array(stats['z_mean'])
        wrapper.z_std[...] = jnp.array(stats['z_std'])

        if obs_type != 'image' and 'state_mean' in stats:
            wrapper.state_mean[...] = jnp.array(stats['state_mean'])
            wrapper.state_std[...] = jnp.array(stats['state_std'])

        # recreate optimizers
        wrapper.optimizer = nnx.Optimizer(wrapper.net, optax.adam(args.prior_step_size), wrt=nnx.Param)
        if obs_type == 'image':
            wrapper.encoder_optimizer = nnx.Optimizer(wrapper.encoder, optax.adam(args.prior_step_size), wrt=nnx.Param)

        return wrapper
