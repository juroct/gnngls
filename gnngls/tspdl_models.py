"""
GNN models for the Traveling Salesperson Problem with Draft Limits (TSPDL).

This module extends the GNNGLS models to handle TSPDL-specific features.
"""

import torch
import torch.nn as nn
import dgl
from typing import Optional, Tuple


class TSPDLEdgeModel(nn.Module):
    """
    Edge property prediction model for TSPDL.

    This model predicts edge properties (e.g., whether an edge should be in the tour)
    based on node features (coordinates, demand, draft limit) and edge features.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        n_layers: int = 3,
        n_heads: int = 8
    ):
        """
        Initialize the model.

        Args:
            in_dim: Input feature dimension
            embed_dim: Embedding dimension
            out_dim: Output dimension
            n_layers: Number of GNN layers
            n_heads: Number of attention heads
        """
        super(TSPDLEdgeModel, self).__init__()

        # Input embedding
        self.edge_embedding = nn.Linear(in_dim, embed_dim)
        self.node_embedding = nn.Linear(4, embed_dim)  # [x, y, demand, draft_limit]

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            dgl.nn.GraphConv(embed_dim, embed_dim, norm='both', weight=True, bias=True)
            for _ in range(n_layers)
        ])

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        edge_feats: torch.Tensor,
        node_feats: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            g: DGL graph
            edge_feats: Edge features
            node_feats: Node features (optional)

        Returns:
            edge_preds: Edge predictions
        """
        # Embed edge features
        h_e = self.edge_embedding(edge_feats)
        g.edata['h'] = h_e

        # Embed node features if provided
        if node_feats is not None:
            h_v = self.node_embedding(node_feats)
            g.ndata['h'] = h_v

        # Apply GNN layers
        h = g.ndata['h']
        for layer in self.gnn_layers:
            h = layer(g, h)
            h = torch.relu(h)
        g.ndata['h'] = h

        # Get edge embeddings
        g.apply_edges(lambda edges: {'h': edges.src['h'] + edges.dst['h']})
        edge_embeds = g.edata['h']

        # Apply output layer
        edge_preds = self.output(edge_embeds)

        return edge_preds


class TSPDLNodeModel(nn.Module):
    """
    Node property prediction model for TSPDL.

    This model predicts node properties (e.g., whether a node should be visited next)
    based on node features (coordinates, demand, draft limit).
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        n_layers: int = 3,
        n_heads: int = 8
    ):
        """
        Initialize the model.

        Args:
            in_dim: Input feature dimension
            embed_dim: Embedding dimension
            out_dim: Output dimension
            n_layers: Number of GNN layers
            n_heads: Number of attention heads
        """
        super(TSPDLNodeModel, self).__init__()

        # Input embedding
        self.node_embedding = nn.Linear(in_dim, embed_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            dgl.nn.GraphConv(embed_dim, embed_dim, norm='both', weight=True, bias=True)
            for _ in range(n_layers)
        ])

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            g: DGL graph
            node_feats: Node features

        Returns:
            node_preds: Node predictions
        """
        # Embed node features
        h = self.node_embedding(node_feats)
        g.ndata['h'] = h

        # Apply GNN layers
        for layer in self.gnn_layers:
            h = layer(g, h)
            h = torch.relu(h)
        g.ndata['h'] = h

        # Apply output layer
        node_preds = self.output(h)

        return node_preds