import numpy as np
import jax
import jax.numpy as jnp
from sklearn.cluster import KMeans

from params import EvalArgs
from envs import RoomEnv
from allo import ALLOProcessor


class SpectralClustering:
    """spectral clustering using learned ALLO eigenvectors"""
    def __init__(self, env: RoomEnv, processor: ALLOProcessor, args: EvalArgs, save_dir: str, key: jax.random.PRNGKey):
        self.env = env
        self.processor = processor
        self.args = args
        self.save_dir = save_dir
        self.rng_key = key
        
        self.kmeans = None
        self.eigenspace_centroids = None
        self.num_clusters = None

    def perform_clustering(self, eigenspace_representations: jnp.ndarray):
        """perform spectral clustering in given eigenspace"""
        eigenspace = np.array(eigenspace_representations)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        seed = int(jax.random.randint(subkey, (), 0, 2**31 - 1))

        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=seed, n_init=20)
        cluster_labels = self.kmeans.fit_predict(eigenspace)
        return cluster_labels
    
    def calculate_eigenspace_centroids(self, eigenspace_representations: jnp.ndarray, cluster_labels: jnp.ndarray):
        """calculate cluster centroids in eigenspace"""
        unique_labels = np.unique(cluster_labels)
        eigenspace_centroids = []

        for label in unique_labels:
            mask = cluster_labels == label
            cluster_observations = eigenspace_representations[mask]

            # compute mean
            centroid = np.mean(cluster_observations, axis=0)

            # find medoid
            distances = np.linalg.norm(cluster_observations - centroid, axis=1)
            medoid_index = np.argmin(distances)
            medoid = cluster_observations[medoid_index]
            eigenspace_centroids.append(medoid)

        return jnp.array(eigenspace_centroids)
    
    def perform_spectral_clustering(self, eval_obs: np.ndarray, num_clusters: int):
        """perform spectral clustering"""
        # get number of clusters
        self.num_clusters = num_clusters

        # project observations to eigenspace
        eigenspace_representations = self.processor.observations_to_eigenspace(eval_obs)
        
        # perform clustering for given number of clusters
        cluster_labels = self.perform_clustering(eigenspace_representations)
        
        # compute eigenspace centroids
        self.eigenspace_centroids = self.calculate_eigenspace_centroids(eigenspace_representations, cluster_labels)

    def predict_cluster_for_observation(self, observation: np.ndarray):
        """predict cluster for an observation"""
        if self.kmeans is None:
            raise ValueError("Must call perform_spectral_clustering() first")
        
        # get eigenspace representation
        eigenspace_representation = self.processor.observation_to_eigenspace(observation)
        eigenspace_representation = np.array(eigenspace_representation)

        if eigenspace_representation.ndim == 1:
            eigenspace_representation = eigenspace_representation[np.newaxis, :]
        
        # predict cluster
        cluster_id = self.kmeans.predict(eigenspace_representation)[0]
        return cluster_id
    
    def predict_clusters_for_observations(self, observations: np.ndarray):
        """predict clusters for multiple observations"""
        if self.kmeans is None:
            raise ValueError("Must call perform_spectral_clustering() first")
        
        # get eigenspace representations
        eigenspace_representations = self.processor.observations_to_eigenspace(observations)
        eigenspace_representations = np.array(eigenspace_representations)
        
        # predict clusters
        cluster_ids = self.kmeans.predict(eigenspace_representations)
        return cluster_ids
