"""
Algorithms for solving the Traveling Salesperson Problem with Draft Limits (TSPDL).

This module extends the GNNGLS algorithms to handle TSPDL constraints.
"""

import time
import numpy as np
import torch
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union, Any

from .tspdl import TSPDLInstance, TSPDLSolution
from .operators import two_opt_a2a, relocate_a2a, two_opt_o2a, relocate_o2a


def nearest_neighbor_tspdl(
    instance: TSPDLInstance,
    start_node: int = 0
) -> List[int]:
    """
    Nearest neighbor algorithm for TSPDL.

    Args:
        instance: TSPDL instance
        start_node: Starting node (depot)

    Returns:
        tour: List of node indices representing the tour
    """
    problem_size = instance.problem_size
    node_xy = instance.node_xy
    node_demand = instance.node_demand
    node_draft_limit = instance.node_draft_limit

    # Initialize tour with start node
    tour = [start_node]

    # Initialize current load
    current_load = 0.0

    # Create a set of unvisited nodes
    unvisited = set(range(problem_size))
    unvisited.remove(start_node)

    # Current node is the start node
    current_node = start_node

    # While there are unvisited nodes
    while unvisited:
        # Find the nearest unvisited node that respects draft limit
        min_dist = float('inf')
        nearest_node = None

        for node in unvisited:
            # Check if draft limit is respected
            if current_load > node_draft_limit[node]:
                continue

            # Calculate distance
            x1, y1 = node_xy[current_node]
            x2, y2 = node_xy[node]
            dist = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            # Update nearest node if closer
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        # If no feasible node is found, break
        if nearest_node is None:
            break

        # Add nearest node to tour
        tour.append(nearest_node)

        # Update current node and load
        current_node = nearest_node
        current_load += node_demand[nearest_node]

        # Remove from unvisited
        unvisited.remove(nearest_node)

    # Return to depot
    if tour[-1] != start_node:
        tour.append(start_node)

    return tour


def insertion_tspdl(
    instance: TSPDLInstance,
    start_node: int = 0
) -> List[int]:
    """
    Insertion algorithm for TSPDL.

    Args:
        instance: TSPDL instance
        start_node: Starting node (depot)

    Returns:
        tour: List of node indices representing the tour
    """
    problem_size = instance.problem_size
    node_xy = instance.node_xy
    node_demand = instance.node_demand
    node_draft_limit = instance.node_draft_limit

    # Initialize tour with start node (depot loop)
    tour = [start_node, start_node]

    # Initialize loads at each position
    loads = [0.0, 0.0]

    # Create a set of unvisited nodes
    unvisited = set(range(problem_size))
    unvisited.remove(start_node)

    # While there are unvisited nodes
    while unvisited:
        # Find the best insertion
        min_cost_increase = float('inf')
        best_insertion = None
        best_node = None

        for node in unvisited:
            # Try all possible insertion positions
            for i in range(1, len(tour)):
                # Check if draft limit is respected
                if loads[i-1] > node_draft_limit[node]:
                    continue

                # Calculate cost increase
                prev_node = tour[i-1]
                next_node = tour[i]

                # Distance before insertion
                x1, y1 = node_xy[prev_node]
                x2, y2 = node_xy[next_node]
                dist_before = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                # Distance after insertion
                x3, y3 = node_xy[node]
                dist_after = torch.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2) + \
                             torch.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

                # Cost increase
                cost_increase = dist_after - dist_before

                # Update best insertion if better
                if cost_increase < min_cost_increase:
                    min_cost_increase = cost_increase
                    best_insertion = i
                    best_node = node

        # If no feasible insertion is found, break
        if best_insertion is None:
            break

        # Insert best node
        tour.insert(best_insertion, best_node)

        # Update loads
        new_loads = loads.copy()
        new_loads.insert(best_insertion, loads[best_insertion-1] + node_demand[best_node])

        # Update loads after insertion
        for i in range(best_insertion + 1, len(new_loads)):
            new_loads[i] += node_demand[best_node]

        loads = new_loads

        # Remove from unvisited
        unvisited.remove(best_node)

    return tour


def local_search_tspdl(
    instance: TSPDLInstance,
    initial_tour: List[int],
    max_iterations: int = 1000,
    first_improvement: bool = False
) -> Tuple[List[int], float, int]:
    """
    Local search algorithm for TSPDL.

    Args:
        instance: TSPDL instance
        initial_tour: Initial tour
        max_iterations: Maximum number of iterations
        first_improvement: Whether to use first improvement strategy

    Returns:
        tour: Improved tour
        cost: Cost of the improved tour
        iterations: Number of iterations performed
    """
    # Convert instance to NetworkX graph for compatibility with operators
    G = instance.to_networkx()

    # Get distance matrix
    edge_weight, _ = nx.attr_matrix(G, 'weight')

    # Initialize tour and cost
    tour = initial_tour.copy()
    cost = TSPDLSolution(instance, tour).cost

    # Initialize iteration counter
    iterations = 0

    # Local search loop
    while iterations < max_iterations:
        iterations += 1

        # Try 2-opt moves
        delta, new_tour = two_opt_a2a(tour, edge_weight, first_improvement)

        # Check if the new tour is feasible
        if delta < 0:
            solution = TSPDLSolution(instance, new_tour)
            if solution.feasible:
                tour = new_tour
                cost += delta
                continue

        # Try relocate moves
        delta, new_tour = relocate_a2a(tour, edge_weight, first_improvement)

        # Check if the new tour is feasible
        if delta < 0:
            solution = TSPDLSolution(instance, new_tour)
            if solution.feasible:
                tour = new_tour
                cost += delta
                continue

        # If no improvement, break
        break

    return tour, cost, iterations


def guided_local_search_tspdl(
    instance: TSPDLInstance,
    initial_tour: List[int],
    time_limit: float,
    lambda_factor: float = 0.1,
    perturbation_moves: int = 5,
    first_improvement: bool = False,
    guides: List[str] = ['weight'],
    track_progress: bool = False,
    max_iterations: int = 100  # Add maximum iterations as a safety measure
) -> Tuple[List[int], float, Dict]:
    """
    Guided local search algorithm for TSPDL.

    Args:
        instance: TSPDL instance
        initial_tour: Initial tour
        time_limit: Time limit in seconds
        lambda_factor: Penalty factor
        perturbation_moves: Number of perturbation moves
        first_improvement: Whether to use first improvement strategy
        guides: List of edge attributes to use as guides
        track_progress: Whether to track progress
        max_iterations: Maximum number of iterations

    Returns:
        tour: Best tour found
        cost: Cost of the best tour
        info: Dictionary with information about the search process
    """
    # Convert instance to NetworkX graph for compatibility with operators
    G = instance.to_networkx()

    # Add penalty attribute to edges
    nx.set_edge_attributes(G, 0, 'penalty')

    # Get distance matrix
    edge_weight, _ = nx.attr_matrix(G, 'weight')

    # Calculate penalty factor
    solution = TSPDLSolution(instance, initial_tour)
    k = lambda_factor * solution.cost / instance.problem_size

    # Initialize tour and cost
    cur_tour = initial_tour.copy()
    cur_cost = solution.cost
    cur_feasible = solution.feasible

    # Apply local search to get a good starting point
    ls_tour, ls_cost, _ = local_search_tspdl(instance, cur_tour, first_improvement=first_improvement)
    if TSPDLSolution(instance, ls_tour).feasible:
        cur_tour = ls_tour
        cur_cost = ls_cost
        cur_feasible = True

    # Best tour found so far
    best_tour = cur_tour.copy()
    best_cost = cur_cost
    best_feasible = cur_feasible

    # For tracking progress
    if track_progress:
        progress = {
            'iterations': [],
            'time': [],
            'cost': [],
            'feasible': [],
            'out_of_limit': [],
            'infeasible_nodes': []
        }
    else:
        progress = {}

    # Main GLS loop
    iter_i = 0
    start_time = time.time()

    while time.time() < time_limit and iter_i < max_iterations:
        # Update progress
        if track_progress:
            progress['iterations'].append(iter_i)
            progress['time'].append(time.time() - start_time)
            progress['cost'].append(cur_cost)
            
            # Check feasibility
            solution = TSPDLSolution(instance, cur_tour)
            progress['feasible'].append(solution.feasible)
            progress['out_of_limit'].append(solution.total_out_of_draft_limit())
            progress['infeasible_nodes'].append(solution.count_out_of_draft_limit_nodes())

        # Select guide for this iteration
        guide = guides[iter_i % len(guides)]

        # Penalize edges
        edge_penalties, _ = nx.attr_matrix(G, 'penalty')
        edge_weight_guided = edge_weight + k * edge_penalties

        # Perturbation phase
        moves = 0
        while moves < perturbation_moves:
            # Find edge with maximum utility to penalize
            max_util = -float('inf')
            max_util_e = None

            for i, j in zip(cur_tour[:-1], cur_tour[1:]):
                if G.has_edge(i, j):
                    e = (i, j)
                else:
                    e = (j, i)

                # Calculate utility based on edge weight and penalty
                if 'penalty' in G.edges[e]:
                    penalty = G.edges[e]['penalty']
                else:
                    penalty = 0

                if guide in G.edges[e]:
                    util = G.edges[e][guide] / (1 + penalty)

                    if util > max_util:
                        max_util = util
                        max_util_e = e

            # If no edge found, break
            if max_util_e is None:
                break

            # Penalize edge
            G.edges[max_util_e]['penalty'] = G.edges[max_util_e].get('penalty', 0) + 1.0

            # Update guided edge weights
            edge_penalties, _ = nx.attr_matrix(G, 'penalty')
            edge_weight_guided = edge_weight + k * edge_penalties

            # Apply operators to nodes in the penalized edge
            for n in max_util_e:
                if n != 0:  # Skip depot
                    i = cur_tour.index(n)

                    # Try 2-opt move
                    delta, new_tour = two_opt_o2a(cur_tour, edge_weight_guided, i, first_improvement)
                    if delta < 0:
                        solution = TSPDLSolution(instance, new_tour)
                        if solution.feasible or not cur_feasible:
                            cur_tour = new_tour
                            cur_cost = solution.cost
                            cur_feasible = solution.feasible
                            moves += 1
                            continue

                    # Try relocate move
                    delta, new_tour = relocate_o2a(cur_tour, edge_weight_guided, i, first_improvement)
                    if delta < 0:
                        solution = TSPDLSolution(instance, new_tour)
                        if solution.feasible or not cur_feasible:
                            cur_tour = new_tour
                            cur_cost = solution.cost
                            cur_feasible = solution.feasible
                            moves += 1
                            continue

                    # Try random move if no improvement
                    j = np.random.randint(1, len(cur_tour) - 1)
                    if j != i:
                        new_tour = cur_tour.copy()
                        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                        solution = TSPDLSolution(instance, new_tour)
                        if solution.feasible or not cur_feasible:
                            cur_tour = new_tour
                            cur_cost = solution.cost
                            cur_feasible = solution.feasible
                            moves += 1

            # Increment moves to avoid infinite loop
            moves += 1

        # Intensification phase with local search
        ls_tour, ls_cost, _ = local_search_tspdl(instance, cur_tour, first_improvement=first_improvement)
        solution = TSPDLSolution(instance, ls_tour)
        if solution.feasible or not cur_feasible:
            cur_tour = ls_tour
            cur_cost = ls_cost
            cur_feasible = solution.feasible

        # Update best solution
        if cur_feasible and (not best_feasible or cur_cost < best_cost):
            best_tour = cur_tour.copy()
            best_cost = cur_cost
            best_feasible = True
        elif not best_feasible and cur_cost < best_cost:
            best_tour = cur_tour.copy()
            best_cost = cur_cost
            best_feasible = cur_feasible

        # Increment iteration counter
        iter_i += 1

    # Final update to progress
    if track_progress:
        progress['iterations'].append(iter_i)
        progress['time'].append(time.time() - start_time)
        progress['cost'].append(cur_cost)
        
        # Check feasibility
        solution = TSPDLSolution(instance, cur_tour)
        progress['feasible'].append(solution.feasible)
        progress['out_of_limit'].append(solution.total_out_of_draft_limit())
        progress['infeasible_nodes'].append(solution.count_out_of_draft_limit_nodes())

    return best_tour, best_cost, progress


def reinforcement_learning_tspdl(
    instance: TSPDLInstance,
    model,
    time_limit: float,
    lambda_factor: float = 0.1,
    perturbation_moves: int = 5,
    first_improvement: bool = False,
    track_progress: bool = False
) -> Tuple[List[int], float, Dict]:
    """
    Reinforcement learning algorithm for TSPDL.

    Args:
        instance: TSPDL instance
        model: RL model
        time_limit: Time limit in seconds
        lambda_factor: Penalty factor
        perturbation_moves: Number of perturbation moves
        first_improvement: Whether to use first improvement strategy
        track_progress: Whether to track progress

    Returns:
        tour: Best tour found
        cost: Cost of the best tour
        info: Dictionary with information about the search process
    """
    # This is a placeholder for a more sophisticated RL-based algorithm
    # For now, we just use the guided local search algorithm
    return guided_local_search_tspdl(
        instance,
        nearest_neighbor_tspdl(instance),
        time_limit,
        lambda_factor,
        perturbation_moves,
        first_improvement,
        track_progress=track_progress
    )