"""
GNN-guided local search for TSPDL.

This module provides functions for GNN-guided local search for TSPDL.
"""

import time
import torch
import dgl
import numpy as np
from typing import List, Tuple, Optional, Dict

from .tspdl import TSPDLInstance, TSPDLSolution
from .tspdl_models import TSPDLEdgeModel
from .tspdl_algorithms import local_search_tspdl


def create_graph_from_instance(
    instance: TSPDLInstance,
    tour: Optional[List[int]] = None
) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a graph from a TSPDL instance.

    Args:
        instance: TSPDL instance
        tour: Tour to use for edge labels (optional)

    Returns:
        g: DGL graph
        edge_features: Edge features
        node_features: Node features
        labels: Edge labels (1 if edge is in tour, 0 otherwise)
    """
    problem_size = instance.problem_size
    device = instance.device
    
    # Create a complete graph
    g = dgl.graph(
        (
            torch.repeat_interleave(torch.arange(problem_size), problem_size - 1),
            torch.cat([
                torch.cat([torch.arange(i), torch.arange(i+1, problem_size)])
                for i in range(problem_size)
            ])
        ),
        device=device
    )
    
    # Create node features: [x, y, demand, draft_limit]
    node_features = torch.cat([
        instance.node_xy,
        instance.node_demand.view(-1, 1),
        instance.node_draft_limit.view(-1, 1)
    ], dim=1)
    
    # Create edge features
    # We'll use [distance, load_difference, draft_limit_difference]
    # First, create distance feature
    src, dst = g.edges()
    distances = torch.sqrt(
        (instance.node_xy[src, 0] - instance.node_xy[dst, 0]) ** 2 +
        (instance.node_xy[src, 1] - instance.node_xy[dst, 1]) ** 2
    ).view(-1, 1)
    
    # Create load feature
    loads = instance.node_demand[src].view(-1, 1)
    
    # Create draft limit feature
    draft_limit_diff = instance.node_draft_limit[dst] - loads
    draft_limit_diff = draft_limit_diff.view(-1, 1)
    
    # Combine features
    edge_features = torch.cat([
        distances,
        loads,
        draft_limit_diff
    ], dim=1)
    
    # Create labels if tour is provided
    if tour is not None:
        # Convert tour to edge list
        tour_edges = set()
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i+1]
            tour_edges.add((min(u, v), max(u, v)))
        
        # Create labels
        labels = torch.zeros(g.number_of_edges(), device=device)
        for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
            if (min(u, v), max(u, v)) in tour_edges:
                labels[i] = 1.0
    else:
        # If no tour is provided, just return zeros
        labels = torch.zeros(g.number_of_edges(), device=device)
    
    return g, edge_features, node_features, labels


def update_graph_with_tour(
    g: dgl.DGLGraph,
    tour: List[int]
) -> torch.Tensor:
    """
    Update graph with tour information.

    Args:
        g: DGL graph
        tour: Tour to use for edge labels

    Returns:
        labels: Edge labels (1 if edge is in tour, 0 otherwise)
    """
    device = g.device
    src, dst = g.edges()
    
    # Convert tour to edge list
    tour_edges = set()
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        tour_edges.add((min(u, v), max(u, v)))
    
    # Create labels
    labels = torch.zeros(g.number_of_edges(), device=device)
    for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        if (min(u, v), max(u, v)) in tour_edges:
            labels[i] = 1.0
    
    return labels


def predict_edge_scores(
    model: TSPDLEdgeModel,
    instance: TSPDLInstance,
    tour: Optional[List[int]] = None
) -> Dict[Tuple[int, int], float]:
    """
    Predict edge scores using a GNN model.

    Args:
        model: GNN model
        instance: TSPDL instance
        tour: Current tour (optional)

    Returns:
        edge_scores: Dictionary mapping edge tuples to scores
    """
    device = instance.device
    model.to(device)
    model.eval()
    
    # Create graph
    g, edge_features, node_features, _ = create_graph_from_instance(instance, tour)
    
    # Predict edge scores
    with torch.no_grad():
        edge_preds = model(g, edge_features, node_features)
        edge_scores = torch.sigmoid(edge_preds).squeeze()
    
    # Convert to dictionary
    src, dst = g.edges()
    edge_scores_dict = {}
    for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        edge_scores_dict[(u, v)] = edge_scores[i].item()
    
    return edge_scores_dict


def gnn_guided_local_search_tspdl(
    instance: TSPDLInstance,
    initial_tour: List[int],
    time_limit: float,
    model: Optional[TSPDLEdgeModel] = None,
    penalty_factor: float = 0.3,
    max_steps: int = 1000,
    weights: Optional[Dict[Tuple[int, int], float]] = None
) -> Tuple[List[int], float, int]:
    """
    Apply GNN-guided local search to a TSPDL instance.

    Args:
        instance: TSPDL instance
        initial_tour: Initial tour
        time_limit: Time limit in seconds
        model: GNN model (optional)
        penalty_factor: Penalty factor for guided local search
        max_steps: Maximum number of steps
        weights: Edge weights (optional)

    Returns:
        best_tour: Best tour found
        best_cost: Best cost found
        steps: Number of steps performed
    """
    # Initialize
    current_tour = initial_tour.copy()
    current_solution = TSPDLSolution(instance, current_tour)
    current_cost = current_solution.cost
    
    best_tour = current_tour.copy()
    best_cost = current_cost
    
    # Initialize penalties
    penalties = {}
    for i in range(instance.problem_size):
        for j in range(i+1, instance.problem_size):
            penalties[(i, j)] = 0.0
    
    # Initialize utilities
    utilities = {}
    
    # If model is provided, predict edge scores
    edge_scores = {}
    if model is not None:
        edge_scores = predict_edge_scores(model, instance, current_tour)
    
    # Initialize objective function with penalties
    def penalized_cost(tour):
        solution = TSPDLSolution(instance, tour)
        if not solution.feasible:
            return float('inf')
        
        cost = solution.cost
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i+1]
            min_u, min_v = min(u, v), max(u, v)
            cost += penalty_factor * penalties.get((min_u, min_v), 0.0)
        
        return cost
    
    # Main loop
    steps = 0
    start_time = time.time()
    while steps < max_steps and time.time() < time_limit:
        # Perform local search with penalized cost function
        current_tour, current_cost, _ = local_search_tspdl(
            instance, current_tour, penalized_cost, max_steps=50
        )
        
        # Update best solution if improved
        current_solution = TSPDLSolution(instance, current_tour)
        if current_solution.feasible and current_solution.cost < best_cost:
            best_tour = current_tour.copy()
            best_cost = current_solution.cost
        
        # Update utilities
        for i in range(len(current_tour) - 1):
            u, v = current_tour[i], current_tour[i+1]
            min_u, min_v = min(u, v), max(u, v)
            
            # Get edge length
            u_coords = instance.node_xy[u].cpu().numpy()
            v_coords = instance.node_xy[v].cpu().numpy()
            edge_length = np.sqrt(np.sum((u_coords - v_coords) ** 2))
            
            # Use model predictions if available
            edge_score = edge_scores.get((u, v), 0.5) if model is not None else 0.5
            
            # Calculate utility: length / (1 + penalty)
            utility = edge_length / (1.0 + penalties.get((min_u, min_v), 0.0))
            
            # Adjust utility based on model prediction
            if model is not None:
                # Higher score means more likely to be in optimal tour
                # So we reduce utility for edges with high scores
                utility *= (2.0 - edge_score)
            
            utilities[(min_u, min_v)] = utility
        
        # Find maximum utility
        max_utility = 0.0
        for edge, utility in utilities.items():
            if utility > max_utility:
                max_utility = utility
        
        # Update penalties for edges in current tour
        if max_utility > 0:
            for i in range(len(current_tour) - 1):
                u, v = current_tour[i], current_tour[i+1]
                min_u, min_v = min(u, v), max(u, v)
                
                if utilities.get((min_u, min_v), 0.0) == max_utility:
                    penalties[(min_u, min_v)] = penalties.get((min_u, min_v), 0.0) + 1.0
        
        steps += 1
    
    return best_tour, best_cost, steps