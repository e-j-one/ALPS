import os
import jax

from params import EvalArgs
from envs import RoomEnv
from utils import EnvironmentHelper
from allo import ALLOProcessor
from dynamics import Dynamics
from prior import Prior
from clustering import SpectralClustering
from graph import ClusterGraph
from .hierarchical import HierarchicalPlanner
from .cem import CEMPlanner


class Planner:
    """main planner class that calls hierarchical and cem planners"""
    def __init__(self, env: RoomEnv, env_helper: EnvironmentHelper, processor: ALLOProcessor, dynamics: Dynamics, prior: Prior, spectral_clustering: SpectralClustering,
                 cluster_graph: ClusterGraph, args: EvalArgs, save_dir: str, key: jax.random.PRNGKey, save_video=False):
        self.env = env
        self.env_helper = env_helper
        self.processor = processor
        self.dynamics = dynamics
        self.prior = prior
        self.spectral_clustering = spectral_clustering
        self.cluster_graph = cluster_graph
        self.args = args
        self.save_dir = save_dir
        self.plan_dir = os.path.join(save_dir, "plan")
        os.makedirs(self.plan_dir, exist_ok=True)
        self.save_video = save_video
        
        # initialize planners
        self.hierarchical_planner = None
        self.cem_planner = None

        # split keys for planners
        self.hierarchical_key, self.cem_key = jax.random.split(key)
    
    def get_hierarchical_planner(self):
        """hierarchical planner instance"""
        self.hierarchical_planner = HierarchicalPlanner(self.env, self.env_helper, self.processor, self.dynamics, self.prior, self.spectral_clustering, self.cluster_graph, self.args, self.hierarchical_key, self.plan_dir)
        return self.hierarchical_planner
    
    def get_cem_planner(self):
        """cem planner instance"""
        self.cem_planner = CEMPlanner(self.env, self.env_helper, self.processor, self.dynamics, self.prior, self.args, self.cem_key, self.plan_dir)
        return self.cem_planner
