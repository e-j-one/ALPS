import os
import matplotlib.pyplot as plt
import networkx as nx


class GraphVisualizer:
    """visualization utilities for cluster connectivity graphs"""
    def __init__(self, env, save_dir: str):
        self.env = env
        self.save_dir = save_dir
        self.graph_dir = os.path.join(save_dir, "graph")
        os.makedirs(self.graph_dir, exist_ok=True)

    def visualize_network_graph(self, G: nx.DiGraph, num_clusters: int):
        """visualize only the network graph representation of cluster connectivity"""
        # create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # compute layout
        pos = nx.kamada_kawai_layout(G)
        
        # draw network graph
        nx.draw(G, pos, ax, node_size=1000, node_color='lightblue', with_labels=True, font_size=14, font_weight='bold', arrows=True, arrowsize=20, edge_color='blue', width=2)
        ax.set_title(f'Cluster Network Graph ({num_clusters} clusters)', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        
        if self.save_dir:
            filename = f"{num_clusters}_network_graph.png"
            save_path = os.path.join(self.graph_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
