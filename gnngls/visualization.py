"""
Visualization utilities for GNNGLS.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import torch
from typing import List, Tuple, Dict, Optional, Union, Any

from .solvers import ANY_SOLVER_AVAILABLE
if ANY_SOLVER_AVAILABLE:
    from .solvers import get_optimal_tour


def plot_tsp_instance(
    G: nx.Graph,
    title: str = "TSP Instance",
    node_size: int = 100,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a TSP instance.
    
    Args:
        G: NetworkX graph representing the TSP instance
        title: Plot title
        node_size: Size of nodes in the plot
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to show the figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color='skyblue', 
        node_size=node_size,
        ax=ax
    )
    
    # Draw edges with weights as colors
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Normalize weights for coloring
    if weights:
        min_weight = min(weights)
        max_weight = max(weights)
        norm = plt.Normalize(min_weight, max_weight)
        cmap = cm.viridis
        
        # Draw edges
        for i, (u, v) in enumerate(edges):
            ax.plot(
                [pos[u][0], pos[v][0]], 
                [pos[u][1], pos[v][1]],
                alpha=0.5,
                color=cmap(norm(weights[i])),
                linewidth=1
            )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Edge Weight')
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, 
        font_size=10, 
        font_family='sans-serif',
        ax=ax
    )
    
    # Highlight depot
    if 0 in G.nodes:
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[0], 
            node_color='red', 
            node_size=node_size*1.5,
            ax=ax
        )
    
    # Set title and axis properties
    ax.set_title(title)
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_aspect('equal')
    
    # Adjust limits to include all nodes with some padding
    x_values = [x for x, y in pos.values()]
    y_values = [y for x, y in pos.values()]
    x_margin = (max(x_values) - min(x_values)) * 0.1
    y_margin = (max(y_values) - min(y_values)) * 0.1
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_tour(
    G: nx.Graph,
    tour: List[int],
    title: str = "TSP Tour",
    node_size: int = 100,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True,
    highlight_edges: Optional[List[Tuple[int, int]]] = None
) -> plt.Figure:
    """
    Plot a TSP tour.
    
    Args:
        G: NetworkX graph representing the TSP instance
        tour: List of nodes representing the tour
        title: Plot title
        node_size: Size of nodes in the plot
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to show the figure
        highlight_edges: List of edges to highlight
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color='skyblue', 
        node_size=node_size,
        ax=ax
    )
    
    # Draw tour edges
    tour_edges = list(zip(tour[:-1], tour[1:]))
    
    # Draw tour edges
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=tour_edges, 
        width=2, 
        alpha=0.8, 
        edge_color='blue',
        ax=ax
    )
    
    # Highlight specific edges if provided
    if highlight_edges:
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=highlight_edges, 
            width=3, 
            alpha=1.0, 
            edge_color='red',
            ax=ax
        )
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, 
        font_size=10, 
        font_family='sans-serif',
        ax=ax
    )
    
    # Highlight depot
    if 0 in G.nodes:
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[0], 
            node_color='red', 
            node_size=node_size*1.5,
            ax=ax
        )
    
    # Calculate tour cost
    tour_cost = sum(G[tour[i]][tour[i+1]]['weight'] for i in range(len(tour)-1))
    
    # Set title and axis properties
    ax.set_title(f"{title} (Cost: {tour_cost:.4f})")
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_aspect('equal')
    
    # Add tour order annotations
    for i, node in enumerate(tour[:-1]):  # Skip the last node (same as first)
        ax.annotate(
            f"{i}",
            xy=pos[node],
            xytext=(5, 5),
            textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
        )
    
    # Adjust limits to include all nodes with some padding
    x_values = [x for x, y in pos.values()]
    y_values = [y for x, y in pos.values()]
    x_margin = (max(x_values) - min(x_values)) * 0.1
    y_margin = (max(y_values) - min(y_values)) * 0.1
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_algorithm_comparison(
    G: nx.Graph,
    algorithm_tours: Dict[str, List[int]],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a comparison of different algorithms.
    
    Args:
        G: NetworkX graph representing the TSP instance
        algorithm_tours: Dictionary mapping algorithm names to tours
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to show the figure
        
    Returns:
        fig: Matplotlib figure
    """
    n_algorithms = len(algorithm_tours)
    n_cols = min(3, n_algorithms)
    n_rows = (n_algorithms + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')
    
    # Plot each algorithm's tour
    for i, (algorithm, tour) in enumerate(algorithm_tours.items()):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='skyblue', 
            node_size=100,
            ax=ax
        )
        
        # Draw tour edges
        tour_edges = list(zip(tour[:-1], tour[1:]))
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=tour_edges, 
            width=2, 
            alpha=0.8, 
            edge_color='blue',
            ax=ax
        )
        
        # Highlight depot
        if 0 in G.nodes:
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[0], 
                node_color='red', 
                node_size=150,
                ax=ax
            )
        
        # Calculate tour cost
        tour_cost = sum(G[tour[i]][tour[i+1]]['weight'] for i in range(len(tour)-1))
        
        # Set title and axis properties
        ax.set_title(f"{algorithm} (Cost: {tour_cost:.4f})")
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_aspect('equal')
        
        # Adjust limits to include all nodes with some padding
        x_values = [x for x, y in pos.values()]
        y_values = [y for x, y in pos.values()]
        x_margin = (max(x_values) - min(x_values)) * 0.1
        y_margin = (max(y_values) - min(y_values)) * 0.1
        ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
        ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # Hide empty subplots
    for i in range(n_algorithms, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_gnn_predictions(
    G: nx.Graph,
    edge_scores: torch.Tensor,
    tour: Optional[List[int]] = None,
    title: str = "GNN Edge Predictions",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot GNN edge predictions.
    
    Args:
        G: NetworkX graph representing the TSP instance
        edge_scores: Tensor of edge scores (higher is more likely to be in the tour)
        tour: Optimal tour (optional)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to show the figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color='skyblue', 
        node_size=100,
        ax=ax
    )
    
    # Get edges and scores
    edges = list(G.edges())
    edge_dict = {(u, v): i for i, (u, v) in enumerate(edges)}
    edge_dict.update({(v, u): i for i, (u, v) in enumerate(edges)})  # Add reverse edges
    
    # Normalize scores for coloring
    scores = edge_scores.squeeze().cpu().numpy()
    min_score = np.min(scores)
    max_score = np.max(scores)
    norm = plt.Normalize(min_score, max_score)
    cmap = cm.viridis
    
    # Draw edges
    for u, v in edges:
        idx = edge_dict[(u, v)]
        score = scores[idx]
        ax.plot(
            [pos[u][0], pos[v][0]], 
            [pos[u][1], pos[v][1]],
            alpha=0.5,
            color=cmap(norm(score)),
            linewidth=2
        )
    
    # Draw tour edges if provided
    if tour is not None:
        tour_edges = list(zip(tour[:-1], tour[1:]))
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=tour_edges, 
            width=3, 
            alpha=0.8, 
            edge_color='red',
            style='dashed',
            ax=ax
        )
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, 
        font_size=10, 
        font_family='sans-serif',
        ax=ax
    )
    
    # Highlight depot
    if 0 in G.nodes:
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[0], 
            node_color='red', 
            node_size=150,
            ax=ax
        )
    
    # Set title and axis properties
    ax.set_title(title)
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_aspect('equal')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Edge Score')
    
    # Adjust limits to include all nodes with some padding
    x_values = [x for x, y in pos.values()]
    y_values = [y for x, y in pos.values()]
    x_margin = (max(x_values) - min(x_values)) * 0.1
    y_margin = (max(y_values) - min(y_values)) * 0.1
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_guided_local_search_progress(
    G: nx.Graph,
    tours: List[List[int]],
    costs: List[float],
    penalties: Optional[List[np.ndarray]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    show: bool = True,
    max_tours: int = 4
) -> plt.Figure:
    """
    Plot guided local search progress.
    
    Args:
        G: NetworkX graph representing the TSP instance
        tours: List of tours generated during the search
        costs: List of costs corresponding to the tours
        penalties: List of penalty matrices for each iteration
        figsize: Figure size
        save_path: Path to save the figure
        show: Whether to show the figure
        max_tours: Maximum number of tours to display
        
    Returns:
        fig: Matplotlib figure
    """
    n_iters = len(tours)
    sample_indices = np.linspace(0, n_iters - 1, max_tours, dtype=int)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, max_tours)
    
    # Plot cost curve
    ax_cost = fig.add_subplot(gs[0, :])
    ax_cost.plot(range(n_iters), costs, '-o', markersize=4)
    ax_cost.set_xlabel('Iteration')
    ax_cost.set_ylabel('Tour Cost')
    ax_cost.set_title('Guided Local Search Progress')
    ax_cost.grid(True)
    
    # Calculate optimal cost if solver is available
    if ANY_SOLVER_AVAILABLE and G.number_of_nodes() <= 100:
        try:
            # Get coordinates from graph
            pos = nx.get_node_attributes(G, 'coords')
            coords = [pos[i] for i in range(G.number_of_nodes())]
            
            # Get optimal tour with Concorde
            opt_tour, opt_cost = get_optimal_tour(coords)
            
            # Plot optimal cost
            ax_cost.axhline(y=opt_cost, color='r', linestyle='--', label='Optimal Cost')
            ax_cost.legend()
        except Exception as e:
            print(f"Warning: Could not compute optimal cost: {e}")
    
    # Plot sample tours and penalties
    for i, idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[1, i])
        
        # Get tour at this iteration
        tour = tours[idx]
        cost = costs[idx]
        
        # Draw nodes
        pos = nx.get_node_attributes(G, 'coords')
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='skyblue', 
            node_size=50,
            ax=ax
        )
        
        # Draw tour edges
        tour_edges = list(zip(tour[:-1], tour[1:]))
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=tour_edges, 
            width=1.5, 
            alpha=0.8, 
            edge_color='blue',
            ax=ax
        )
        
        # If penalties are provided, draw penalty thickness
        if penalties is not None and idx < len(penalties):
            penalty_matrix = penalties[idx]
            if penalty_matrix is not None:
                for e_idx, (u, v) in enumerate(G.edges()):
                    penalty = penalty_matrix[u, v]
                    if penalty > 0:
                        ax.plot(
                            [pos[u][0], pos[v][0]], 
                            [pos[u][1], pos[v][1]],
                            alpha=0.7,
                            color='red',
                            linewidth=0.5 + penalty,
                            zorder=10
                        )
        
        # Highlight depot
        if 0 in G.nodes:
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[0], 
                node_color='red', 
                node_size=75,
                ax=ax
            )
        
        # Set title and axis properties
        ax.set_title(f"Iter {idx} (Cost: {cost:.4f})")
        ax.set_axis_off()
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_animation(
    G: nx.Graph,
    tours: List[List[int]],
    costs: List[float],
    output_path: str,
    title: str = "Algorithm Progress",
    fps: int = 2
):
    """
    Create an animation of the algorithm progress.
    
    Args:
        G: NetworkX graph representing the TSP instance
        tours: List of tours generated during the search
        costs: List of costs corresponding to the tours
        output_path: Path to save the animation
        title: Animation title
        fps: Frames per second
    """
    try:
        import matplotlib.animation as animation
    except ImportError:
        print("Error: matplotlib.animation is not available.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get node coordinates
    pos = nx.get_node_attributes(G, 'coords')
    
    def init():
        ax.clear()
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='skyblue', 
            node_size=100,
            ax=ax
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos, 
            font_size=10, 
            font_family='sans-serif',
            ax=ax
        )
        
        # Highlight depot
        if 0 in G.nodes:
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[0], 
                node_color='red', 
                node_size=150,
                ax=ax
            )
        
        # Set axis properties
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_aspect('equal')
        
        # Adjust limits to include all nodes with some padding
        x_values = [x for x, y in pos.values()]
        y_values = [y for x, y in pos.values()]
        x_margin = (max(x_values) - min(x_values)) * 0.1
        y_margin = (max(y_values) - min(y_values)) * 0.1
        ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
        ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
        
        return []
    
    def update(frame):
        ax.clear()
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='skyblue', 
            node_size=100,
            ax=ax
        )
        
        # Draw tour edges
        tour = tours[frame]
        tour_edges = list(zip(tour[:-1], tour[1:]))
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=tour_edges, 
            width=2, 
            alpha=0.8, 
            edge_color='blue',
            ax=ax
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos, 
            font_size=10, 
            font_family='sans-serif',
            ax=ax
        )
        
        # Highlight depot
        if 0 in G.nodes:
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[0], 
                node_color='red', 
                node_size=150,
                ax=ax
            )
        
        # Set title and axis properties
        cost = costs[frame]
        ax.set_title(f"{title} - Iteration {frame} (Cost: {cost:.4f})")
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_aspect('equal')
        
        # Adjust limits to include all nodes with some padding
        x_values = [x for x, y in pos.values()]
        y_values = [y for x, y in pos.values()]
        x_margin = (max(x_values) - min(x_values)) * 0.1
        y_margin = (max(y_values) - min(y_values)) * 0.1
        ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
        ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
        
        return []
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(tours), init_func=init, blit=True)
    
    # Save animation
    anim.save(output_path, fps=fps, dpi=100, writer='ffmpeg')
    
    plt.close(fig)