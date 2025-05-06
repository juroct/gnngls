"""
Dataset classes for TSPDL.

This module provides dataset classes for TSPDL instances.
"""

import os
import torch
import pickle
from typing import Tuple, List, Optional

from .tspdl import TSPDLInstance
from .solvers import solve_tsp_lkh, SOLVERS_AVAILABLE


class TSPDLDataset(torch.utils.data.Dataset):
    """
    Dataset for TSPDL instances.

    This class loads TSPDL instances from disk and provides them
    along with their optimal (or near-optimal) solutions.
    """

    def __init__(
        self,
        data_dir: str,
        device: torch.device = None
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the instances
            device: Device to use for tensor operations
        """
        self.data_dir = data_dir
        self.device = device if device is not None else torch.device('cpu')
        
        # Load the list of instances
        list_file = os.path.join(data_dir, 'train.txt') if os.path.exists(os.path.join(data_dir, 'train.txt')) else \
                    os.path.join(data_dir, 'val.txt') if os.path.exists(os.path.join(data_dir, 'val.txt')) else \
                    os.path.join(data_dir, 'test.txt') if os.path.exists(os.path.join(data_dir, 'test.txt')) else \
                    os.path.join(data_dir, 'instances.txt')
        
        with open(list_file, 'r') as f:
            lines = f.readlines()
        
        # Skip the first line if it contains metadata
        start_idx = 1 if not lines[0].endswith('.pkl\n') else 0
        
        # Extract instance file names
        self.instance_files = [
            os.path.join(data_dir, line.strip())
            for line in lines[start_idx:]
        ]
        
        # Load all instances from a single pickle file if that's the format
        if 'instances.pkl' in self.instance_files[0]:
            with open(self.instance_files[0], 'rb') as f:
                instances_data = pickle.load(f)
            
            self.instances = []
            for node_xy, node_demand, node_draft_limit in instances_data:
                self.instances.append(
                    TSPDLInstance(
                        torch.tensor(node_xy, dtype=torch.float32),
                        torch.tensor(node_demand, dtype=torch.float32),
                        torch.tensor(node_draft_limit, dtype=torch.float32),
                        self.device
                    )
                )
        else:
            self.instances = []
            for instance_file in self.instance_files:
                with open(instance_file, 'rb') as f:
                    instance_data = pickle.load(f)
                
                node_xy, node_demand, node_draft_limit = instance_data
                self.instances.append(
                    TSPDLInstance(
                        torch.tensor(node_xy, dtype=torch.float32),
                        torch.tensor(node_demand, dtype=torch.float32),
                        torch.tensor(node_draft_limit, dtype=torch.float32),
                        self.device
                    )
                )
        
        # Precompute optimal tours if solvers are available
        self.optimal_tours = []
        for instance in self.instances:
            if SOLVERS_AVAILABLE:
                # Get node coordinates
                node_xy = instance.node_xy.cpu().numpy()
                
                # Solve using LKH
                tour, _ = solve_tsp_lkh(node_xy)
                
                # LKH returns a tour starting from 0 and ending at 0
                # We need to add 0 at the beginning to match our format
                if tour[0] != 0:
                    tour = [0] + tour
                
                # Add the tour
                self.optimal_tours.append(tour)
            else:
                # If solvers are not available, use a simple heuristic
                tour = list(range(instance.problem_size)) + [0]
                self.optimal_tours.append(tour)
    
    def __len__(self) -> int:
        """
        Get the number of instances in the dataset.

        Returns:
            length: Number of instances
        """
        return len(self.instances)
    
    def __getitem__(self, idx: int) -> Tuple[TSPDLInstance, List[int]]:
        """
        Get an instance and its optimal tour.

        Args:
            idx: Index of the instance

        Returns:
            instance: TSPDL instance
            optimal_tour: Optimal tour for the instance
        """
        return self.instances[idx], self.optimal_tours[idx]