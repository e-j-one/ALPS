import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Union
import gymnasium as gym

from params import Args
from envs import RoomEnv
from utils import Buffer
from clustering import SpectralClustering
from .visualizer import GraphVisualizer


class ClusterGraph:
    """generate cluster connectivity graph"""
    def __init__(self, env: Union[RoomEnv, gym.Env], clustering: SpectralClustering, args: Args, save_dir: str):
        self.env = env
        self.spectral_clustering = clustering
        self.args = args
        self.save_dir = save_dir

        # store connectivity graph
        self.G = None
        
        # visualizer for cluster graphs
        if self.args.show_graph:
            self.visualizer = GraphVisualizer(env, save_dir)
    
    def _apply_top_p_filtering(self, connection_counts: dict, p: float = 0.95) -> set:
        """apply top-p filtering to removes noisy or rare transitions while preserving the main connectivity structure"""
        # group transitions by source cluster
        state_transitions = defaultdict(list)
        for (src, dst), count in connection_counts.items():
            state_transitions[src].append((dst, count))
        
        filtered_connections = set()
        
        for src_state, transitions in state_transitions.items():
            if not transitions:
                continue
            
            # sort by frequency (most common first)
            transitions.sort(key=lambda x: x[1], reverse=True)
            total_count = sum(count for _, count in transitions)
            
            # keep transitions until we reach p% of total probability
            cumulative_prob = 0.0
            for dst_state, count in transitions:
                prob = count / total_count
                cumulative_prob += prob
                filtered_connections.add((src_state, dst_state))
                
                if cumulative_prob >= p:
                    break
        
        return filtered_connections
    
    def create_cluster_graph(self, buffer: Buffer, top_p: float):
        """create cluster connectivity graph from replay buffer"""
        # get valid transition indices
        valid_indices = buffer.valid_indices[:buffer.valid_count]
        num_transitions = len(valid_indices)
        batch_size = self.args.batch_size

        # predict clusters and count transitions
        connection_counts = defaultdict(int)

        for i in range(0, num_transitions, batch_size):
            end = min(i + batch_size, num_transitions)
            batch_idx = valid_indices[i:end]

            batch_current = buffer.observations[batch_idx]
            batch_next = buffer.observations[batch_idx + 1]

            current_clusters = self.spectral_clustering.predict_clusters_for_observations(batch_current)
            next_clusters = self.spectral_clustering.predict_clusters_for_observations(batch_next)

            for src_cluster, dst_cluster in zip(current_clusters, next_clusters):
                if src_cluster != dst_cluster:
                    connection_counts[(src_cluster, dst_cluster)] += 1
        
        # apply top-p filtering to keep main connectivity structure
        filtered_connections = self._apply_top_p_filtering(connection_counts, p=top_p)
        
        # create directed graph
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.spectral_clustering.num_clusters))
        self.G.add_edges_from(filtered_connections)

        if self.args.show_graph:
            self.visualizer.visualize_network_graph(self.G, self.args.num_clusters)
