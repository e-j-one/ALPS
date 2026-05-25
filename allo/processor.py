import numpy as np
import jax.numpy as jnp

from params import TrainArgs
from utils.buffer import normalize_obs
from .model import ALLO


class ALLOProcessor:
    """process data through ALLO to get eigenspace representations"""
    def __init__(self, allo: ALLO, args: TrainArgs):
        self.allo = allo
        self.args = args
    
    def get_eigenvalues(self):
        """get eigenvalues for plotting"""
        return np.array(self.allo.get_eigenvalue_estimates())
    
    def plot_eigenvalues(self, save_dir: str = None):
        """plot eigenvalues"""
        eigenvalues = self.get_eigenvalues()
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(np.arange(0, len(eigenvalues)), eigenvalues, marker='o')
        plt.xlabel('Eigenvector Index')
        plt.ylabel('Eigenvalue')
        plt.title('ALLO Eigenvalues')
        plt.grid()
        if save_dir is not None:
            plt.savefig(f"{save_dir}/eigenvalues.png")
        plt.close()

    def observation_to_eigenspace(self, observation: np.ndarray) -> jnp.ndarray:
        """convert observation to eigenspace representation using ALLO"""
        # normalize observation
        observation = normalize_obs(observation)
        eigenspace_representation = self.allo.get_representations_batch(observation[jnp.newaxis, :], self.args.use_scaling, self.args.skip_first_eigenvector)
        return eigenspace_representation[0]

    def observations_to_eigenspace(self, observations: np.ndarray) -> jnp.ndarray:
        """convert observations to eigenspace representations using ALLO"""
        batch_size = self.args.batch_size
        total_observations = len(observations)

        # determine z_dim from args
        z_dim = self.args.num_eigenvectors if self.args.skip_first_eigenvector else self.args.num_eigenvectors + 1
        eigenspace_representations = jnp.zeros((total_observations, z_dim), dtype=jnp.float32)

        for i in range(0, total_observations, batch_size):
            end_idx = min(i + batch_size, total_observations)
            batch_obs = observations[i:end_idx]
            actual_batch_size = end_idx - i

            # normalize batch
            batch_obs = normalize_obs(batch_obs)

            if actual_batch_size < batch_size:
                padding = jnp.tile(batch_obs[-1:], (batch_size - actual_batch_size,) + (1,) * (batch_obs.ndim - 1))
                batch_obs = jnp.concatenate([batch_obs, padding], axis=0)

            batch_repr = self.allo.get_representations_batch(batch_obs, self.args.use_scaling, self.args.skip_first_eigenvector)
            eigenspace_representations = eigenspace_representations.at[i:end_idx].set(batch_repr[:actual_batch_size])

        return eigenspace_representations

    def centroids_to_eigenspace(self, centroids: np.ndarray) -> jnp.ndarray:
        """convert centroids to eigenspace representations"""
        # normalize centroids
        centroids = normalize_obs(centroids)
        eigenspace_representations = self.allo.get_representations_batch(centroids, self.args.use_scaling, self.args.skip_first_eigenvector)
        return eigenspace_representations
