"""
TSPDL Demo script.

This script demonstrates how to use the GNNGLS framework to solve TSPDL instances.
"""

import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from gnngls import (
    TSPDLInstance, TSPDLSolution,
    nearest_neighbor_tspdl, insertion_tspdl, local_search_tspdl, guided_local_search_tspdl,
    plot_tspdl_instance, plot_tspdl_solution, plot_tspdl_algorithm_comparison, plot_tspdl_gls_progress
)


def parse_args():
    parser = argparse.ArgumentParser(description='TSPDL Demo')
    parser.add_argument('output_dir', type=str, help='Output directory for plots')
    parser.add_argument('--problem_size', type=int, default=20, help='Problem size')
    parser.add_argument('--hardness', type=str, default='medium', choices=['easy', 'medium', 'hard'], help='Problem hardness')
    parser.add_argument('--time_limit', type=float, default=5.0, help='Time limit for guided local search (in seconds)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate a random TSPDL instance
    print(f"Generating a random TSPDL instance with size {args.problem_size} and hardness {args.hardness}...")
    instance = TSPDLInstance.generate_random(
        problem_size=args.problem_size,
        hardness=args.hardness
    )
    
    # Visualize the instance
    print("Visualizing the instance...")
    fig = plot_tspdl_instance(instance)
    fig.savefig(os.path.join(args.output_dir, 'instance.png'))
    plt.close(fig)
    
    # Solve the instance using different algorithms
    print("Solving the instance using different algorithms...")
    
    # Nearest neighbor
    start_time = time.time()
    nn_tour = nearest_neighbor_tspdl(instance)
    nn_time = time.time() - start_time
    nn_solution = TSPDLSolution(instance, nn_tour)
    print(f"Nearest neighbor solution: cost={nn_solution.cost:.4f}, feasible={nn_solution.feasible}, time={nn_time:.4f}s")
    
    # Insertion
    start_time = time.time()
    ins_tour = insertion_tspdl(instance)
    ins_time = time.time() - start_time
    ins_solution = TSPDLSolution(instance, ins_tour)
    print(f"Insertion solution: cost={ins_solution.cost:.4f}, feasible={ins_solution.feasible}, time={ins_time:.4f}s")
    
    # Local search
    start_time = time.time()
    ls_tour, ls_cost, ls_steps = local_search_tspdl(instance, nn_tour.copy())
    ls_time = time.time() - start_time
    ls_solution = TSPDLSolution(instance, ls_tour)
    print(f"Local search solution: cost={ls_solution.cost:.4f}, feasible={ls_solution.feasible}, steps={ls_steps}, time={ls_time:.4f}s")
    
    # Guided local search
    start_time = time.time()
    time_limit = time.time() + args.time_limit
    gls_tour, gls_cost, gls_steps = guided_local_search_tspdl(instance, ins_tour.copy(), time_limit)
    gls_time = time.time() - start_time
    gls_solution = TSPDLSolution(instance, gls_tour)
    print(f"Guided local search solution: cost={gls_solution.cost:.4f}, feasible={gls_solution.feasible}, steps={gls_steps}, time={gls_time:.4f}s")
    
    # Visualize solutions
    print("Visualizing solutions...")
    
    # Nearest neighbor
    fig = plot_tspdl_solution(nn_solution)
    fig.savefig(os.path.join(args.output_dir, 'nn_solution.png'))
    plt.close(fig)
    
    # Insertion
    fig = plot_tspdl_solution(ins_solution)
    fig.savefig(os.path.join(args.output_dir, 'ins_solution.png'))
    plt.close(fig)
    
    # Local search
    fig = plot_tspdl_solution(ls_solution)
    fig.savefig(os.path.join(args.output_dir, 'ls_solution.png'))
    plt.close(fig)
    
    # Guided local search
    fig = plot_tspdl_solution(gls_solution)
    fig.savefig(os.path.join(args.output_dir, 'gls_solution.png'))
    plt.close(fig)
    
    # Compare algorithms
    print("Comparing algorithms...")
    solutions = {
        'Nearest Neighbor': (nn_solution, nn_time),
        'Insertion': (ins_solution, ins_time),
        'Local Search': (ls_solution, ls_time),
        'Guided Local Search': (gls_solution, gls_time)
    }
    fig = plot_tspdl_algorithm_comparison(solutions)
    fig.savefig(os.path.join(args.output_dir, 'algorithm_comparison.png'))
    plt.close(fig)
    
    print(f"Demo completed. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()