"""
Generate TSPDL instances.

This script generates TSPDL instances and saves them to disk.
"""

import os
import argparse
import pickle
import torch
import numpy as np

from gnngls import TSPDLInstance


def parse_args():
    parser = argparse.ArgumentParser(description='Generate TSPDL instances')
    parser.add_argument('output_dir', type=str, help='Output directory for datasets')
    parser.add_argument('--problem_size', type=int, default=20, help='Problem size')
    parser.add_argument('--n_instances', type=int, default=100, help='Number of instances to generate')
    parser.add_argument('--hardness', type=str, default='medium', choices=['easy', 'medium', 'hard'], help='Problem hardness')
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
    
    # Generate instances
    print(f"Generating {args.n_instances} TSPDL instances with size {args.problem_size} and hardness {args.hardness}...")
    
    instances = []
    for i in range(args.n_instances):
        print(f"Generating instance {i+1}/{args.n_instances}...")
        instance = TSPDLInstance.generate_random(
            problem_size=args.problem_size,
            hardness=args.hardness
        )
        instances.append((
            instance.node_xy.cpu().numpy(),
            instance.node_demand.cpu().numpy(),
            instance.node_draft_limit.cpu().numpy()
        ))
    
    # Save to disk
    instances_file = os.path.join(args.output_dir, 'instances.pkl')
    with open(instances_file, 'wb') as f:
        pickle.dump(instances, f)
    
    # Create instances.txt for convenience
    instances_txt = os.path.join(args.output_dir, 'instances.txt')
    with open(instances_txt, 'w') as f:
        f.write(f"{args.n_instances} instances of size {args.problem_size} with hardness {args.hardness}\n")
        for i in range(args.n_instances):
            f.write(f"instance_{i:04d}.pkl\n")
    
    print(f"Generated {args.n_instances} instances.")
    print(f"Saved to {instances_file} and {instances_txt}.")


if __name__ == '__main__':
    main()