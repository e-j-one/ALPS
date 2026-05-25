from dataclasses import dataclass, asdict
import numpy as np
from typing import Optional, List


@dataclass
class PlanResult:
    """store planning results"""
    start_state: np.ndarray
    goal_state: np.ndarray
    steps: int
    final_goal_distance: float
    success: bool
    planner_type: str
    
    # hierarchical-specific
    start_cluster_id: Optional[int] = None
    goal_cluster_id: Optional[int] = None
    cluster_path: Optional[List[int]] = None
    cluster_trajectory: Optional[List[int]] = None
    
    def to_dict(self):
        return asdict(self)
