import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from typing import Tuple, Dict, Any
import numpy as np
from functools import partial
import pickle

from params import TrainArgs
from utils import Buffer

jax.clear_caches()


class Encoder(nnx.Module):
    def __init__(self, input_dim: int, input_shape: Tuple[int, ...], hidden_dim: int, num_eigenvectors: int,
                 duals_initial_val: float, barrier_initial_val: float, obs_type: str, rngs: nnx.Rngs=None
                ):
        
        self.obs_type = obs_type
        self.input_shape = input_shape
        
        # normalization statistics (only for flat observations)
        if obs_type == 'image':
            self.mean = None
            self.std = None
        else:
            self.mean = nnx.Variable(jnp.zeros(input_dim))
            self.std = nnx.Variable(jnp.ones(input_dim))
        
        # CNN layers: input > convs > flatten > linear > layernorm
        if obs_type == 'image':
            self.conv_layers = nnx.Sequential(*[
                nnx.Conv(input_shape[-1], 16, kernel_size=(8, 8), strides=4, rngs=rngs),
                nnx.relu,
                nnx.Conv(16, 16, kernel_size=(4, 4), strides=2, rngs=rngs),
                nnx.relu,
                nnx.Conv(16, 16, kernel_size=(3, 3), strides=2, rngs=rngs),
                nnx.relu
            ])
            conv_output_size = self._get_conv_output_size(input_shape)
            feature_dim = conv_output_size
        else:
            feature_dim = input_dim
        
        # MLP layers: input > linear > layernorm > tanh > linear > relu > linear > relu > linear > output
        self.mlp_layers = nnx.Sequential(*[
            nnx.Linear(feature_dim, hidden_dim, rngs=rngs),
            nnx.LayerNorm(hidden_dim, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, num_eigenvectors, rngs=rngs)
        ])

        self.dual_variables = nnx.Param(jnp.tril(duals_initial_val * jnp.ones((num_eigenvectors, num_eigenvectors))))
        self.barrier_coefficients = nnx.Param(jnp.array(barrier_initial_val))

    def _get_conv_output_size(self, input_shape: Tuple[int, ...]) -> int:
        """get the output size using dummy forward pass"""
        h, w, c = input_shape
        
        dummy_input = jnp.zeros((1, h, w, c))    # (1, H, W, C)
        dummy_output = self.conv_layers(dummy_input)
        output_shape = dummy_output.shape[1:]                       # (H', W', C')
        flattened_size = int(jnp.prod(jnp.array(output_shape)))
        
        return flattened_size
        
    def __call__(self, x):
        if self.obs_type == 'image':
            # x : (batch, height, width, channels)
            x = self.conv_layers(x)
            x = jnp.reshape(x, (x.shape[0], -1))
        else:
            # normalize raw state coordinates/observations
            x = (x - self.mean.value) / self.std.value

        return self.mlp_layers(x)


def allo_loss_function(encoder: Encoder, observations: jnp.ndarray, next_observations: jnp.ndarray, observations_2: jnp.ndarray, step_size_duals: float) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """ALLO loss function"""
    # compute representations
    phi = encoder(observations)
    phi_2 = encoder(observations_2)
    next_phi = encoder(next_observations)

    # graph drawing loss (spectral objective)
    graph_loss = 0.5 * ((phi - next_phi)**2).mean(0).sum()
    
    # get dimensions
    n = phi.shape[0]
    d = phi.shape[1]
    
    # extract parameters
    dual_variables = encoder.dual_variables
    barrier_coefficients = encoder.barrier_coefficients
    
    # compute orthogonality constraint violations
    inner_product_matrix_1 = jnp.einsum('ij,ik->jk', phi, jax.lax.stop_gradient(phi)) / n
    inner_product_matrix_2 = jnp.einsum('ij,ik->jk', phi_2, jax.lax.stop_gradient(phi_2)) / n
    
    error_matrix_1 = jnp.tril(inner_product_matrix_1 - jnp.eye(d))
    error_matrix_2 = jnp.tril(inner_product_matrix_2 - jnp.eye(d))
    
    # compute sequential learning mask
    error_matrix = 0.5 * (error_matrix_1 + error_matrix_2)
    
    # dual variable loss (Augmented Lagrangian linear penalty)
    dual_loss_pos = (jax.lax.stop_gradient(jnp.asarray(dual_variables)) * error_matrix).sum()
    dual_loss_neg = -step_size_duals * (dual_variables * jax.lax.stop_gradient(error_matrix)).sum()
    
    # barrier loss (quadratic penalty)
    quadratic_error_matrix = error_matrix_1 * error_matrix_2
    barrier_loss_pos = jax.lax.stop_gradient(barrier_coefficients) * quadratic_error_matrix.sum()

    quadratic_error = jnp.clip(quadratic_error_matrix, 0, None).mean()
    barrier_loss_neg = -barrier_coefficients * jax.lax.stop_gradient(quadratic_error)
    
    # total ALLO objective
    allo_loss = graph_loss + dual_loss_pos + barrier_loss_pos + dual_loss_neg + barrier_loss_neg
    
    aux_metrics = {
        'allo_loss': allo_loss,
        'graph_loss': graph_loss,
        'dual_loss_pos': dual_loss_pos,
        'dual_loss_neg': dual_loss_neg,
        'barrier_loss_pos': barrier_loss_pos,
        'barrier_loss_neg': barrier_loss_neg,
    }
    
    return allo_loss, aux_metrics


@partial(nnx.jit, static_argnames=['step_size_duals'])
def allo_update_step(encoder: Encoder, encoder_optimizer: nnx.Optimizer, observations: jnp.ndarray, next_observations: jnp.ndarray,  observations_2: jnp.ndarray, step_size_duals: float) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """single training step"""
    # compute loss and gradients
    (loss, aux_metrics), grads = nnx.value_and_grad(
        allo_loss_function, has_aux=True
    )(encoder, observations, next_observations, observations_2, step_size_duals)
    
    # update encoder parameters using optimizer
    encoder_optimizer.update(encoder, grads)
    
    return loss, aux_metrics


class ALLO(nnx.Module):
    """ALLO trainer class"""
    def __init__(self, input_dim: int, input_shape: Tuple[int, ...], args: TrainArgs, key: jax.random.PRNGKey=None):
        self.args = args
        if key is not None:
            self.rngs = nnx.Rngs(int(key[0]))
        else:
            self.rngs = nnx.Rngs(0)
        
        # initialize state
        self.encoder = Encoder(input_dim, input_shape, args.allo_hidden_dim, args.num_eigenvectors+1, args.duals_initial_val,
                               args.barrier_initial_val, args.obs_type, self.rngs)

        self.encoder_optimizer = nnx.Optimizer(self.encoder, optax.adam(args.allo_step_size), wrt=nnx.Param)

    def compute_stats(self, buffer: Buffer):
        """compute normalization stats from buffer"""
        if self.args.obs_type == 'image':
            return

        all_observations = buffer.observations[:buffer.current_size]
        self.encoder.mean.value = jnp.mean(all_observations, axis=0)
        self.encoder.std.value = jnp.std(all_observations, axis=0) + 1e-8
        print("ALLO Observation Statistics Set")
        
    def train_step(self, observations: jnp.ndarray, next_observations: jnp.ndarray, observations_2: jnp.ndarray) -> Dict[str, Any]:
        """execute one training step"""
        _, metrics = allo_update_step(
            self.encoder, self.encoder_optimizer, 
            observations, next_observations, observations_2,
            self.args.step_size_duals,
        )

        # update dual variables and barrier coefficients
        self.encoder.dual_variables.value = jnp.clip(self.encoder.dual_variables.value, self.args.min_duals, self.args.max_duals)
        self.encoder.barrier_coefficients.value = jnp.clip(self.encoder.barrier_coefficients.value, self.args.min_barrier_coefs, self.args.max_barrier_coefs)
        
        return metrics
    
    @partial(nnx.jit, static_argnames=['use_scaling', 'skip_first'])
    def _get_representations_jit(self, observations: jnp.ndarray, use_scaling: bool = False, skip_first: bool = True):
        representations = self.encoder(observations)
        
        diagonal_duals = jnp.diag(self.encoder.dual_variables.value)
        eigenvalues = -diagonal_duals / 2.0
        
        if skip_first:
            representations = representations[:, 1:]
            eigenvalues = eigenvalues[1:]
        
        if use_scaling:
            epsilon = 1e-8
            eigenvalues = jnp.maximum(eigenvalues, epsilon)
            representations = representations / jnp.sqrt(eigenvalues[jnp.newaxis, :])
            
        return representations
    
    def _get_representations(self, observations: jnp.ndarray, use_scaling: bool = False, skip_first: bool = True):
        """non-JIT version for use inside other JIT-compiled functions"""
        representations = self.encoder(observations)
        
        diagonal_duals = jnp.diag(self.encoder.dual_variables.value)
        eigenvalues = -diagonal_duals / 2.0
        
        if skip_first:
            representations = representations[:, 1:]
            eigenvalues = eigenvalues[1:]
        
        if use_scaling:
            epsilon = 1e-8
            eigenvalues = jnp.maximum(eigenvalues, epsilon)
            representations = representations / jnp.sqrt(eigenvalues[jnp.newaxis, :])

        return representations
    
    def get_representations_batch(self, observations: jnp.ndarray, use_scaling: bool = False, skip_first: bool = True) -> jnp.ndarray:
        representations_jax = self._get_representations_jit(observations, use_scaling, skip_first)
        return representations_jax

    def get_representation_start_pos(self, observations: jnp.ndarray, use_scaling: bool = False, skip_first: bool = True) -> jnp.ndarray:
        representations_jax = self._get_representations_jit(observations[np.newaxis, :], use_scaling, skip_first)
        return representations_jax[0]
    
    def get_eigenvalue_estimates(self) -> jnp.ndarray:
        diagonal_duals = jnp.diag(self.encoder.dual_variables.value)
        return -diagonal_duals / 2.0
    
    def checkpoint(self, filepath: str):
        """save model checkpoint"""
        _, state = nnx.split(self.encoder)
        flat_state = dict(state.flat_state())
        state_dict = {'/'.join(map(str, k)): np.array(v.value) for k, v in flat_state.items()}

        save_data = {
            'state_dict': state_dict,
            'obs_type': self.args.obs_type,
            'config': {
                'num_eigenvectors': self.args.num_eigenvectors,
                'allo_hidden_dim': self.args.allo_hidden_dim,
                'obs_type': self.args.obs_type
            }
        }

        # save normalization stats for flat observations
        if self.args.obs_type != 'image':
            save_data['stats'] = {
                'mean': np.array(self.encoder.mean.value),
                'std': np.array(self.encoder.std.value)
            }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load_checkpoint(cls, filepath: str, input_dim: int, input_shape: Tuple[int, ...], args: TrainArgs):
        """load model from checkpoint"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        state_dict = data['state_dict']
        config = data['config']
        obs_type = data.get('obs_type', 'xy')  # default to 'xy' for backwards compatibility

        print(f"ALLO Model loaded from: {filepath}")
        print(f"  obs_type: {obs_type}")

        if config['num_eigenvectors'] != args.num_eigenvectors:
            print(f" WARNING: Args specify {args.num_eigenvectors} eigenvectors, but model has {config['num_eigenvectors']}.")

        # create fresh instance
        dummy_key = jax.random.PRNGKey(0)
        allo_instance = cls(input_dim, input_shape, args, dummy_key)
        
        # restore state
        _, state = nnx.split(allo_instance.encoder)

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
        
        # restore normalization stats for flat observations
        if obs_type != 'image' and 'stats' in data:
            stats = data['stats']
            allo_instance.encoder.mean.value = jnp.array(stats['mean'])
            allo_instance.encoder.std.value = jnp.array(stats['std'])

        allo_instance.encoder_optimizer = nnx.Optimizer(allo_instance.encoder, optax.adam(args.allo_step_size), wrt=nnx.Param)

        return allo_instance
