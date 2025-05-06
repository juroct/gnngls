"""
Train a GNN model for TSPDL.

This script trains a GNN model for TSPDL using the provided dataset.
"""

import os
import time
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from gnngls import TSPDLDataset, TSPDLEdgeModel, create_graph_from_instance


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GNN model for TSPDL')
    parser.add_argument('data_dir', type=str, help='Data directory containing train and val directories')
    parser.add_argument('output_dir', type=str, help='Output directory for models and logs')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    return parser.parse_args()


def train_epoch(model, loader, optimizer, device):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        # Process each instance in the batch
        batch_loss = 0.0
        for instance, optimal_tour in batch:
            # Move instance to device
            instance = instance.to(device)
            
            # Create graph with optimal tour edges as labels
            g, edge_features, node_features, labels = create_graph_from_instance(instance, optimal_tour)
            g = g.to(device)
            edge_features = edge_features.to(device)
            node_features = node_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            edge_preds = model(g, edge_features, node_features)
            edge_preds = edge_preds.squeeze()
            
            # Compute loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(edge_preds, labels)
            batch_loss += loss
        
        # Average loss over batch and backpropagate
        batch_loss /= len(batch)
        batch_loss.backward()
        optimizer.step()
        
        epoch_loss += batch_loss.item()
        n_batches += 1
    
    return epoch_loss / n_batches


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            # Process each instance in the batch
            batch_loss = 0.0
            for instance, optimal_tour in batch:
                # Move instance to device
                instance = instance.to(device)
                
                # Create graph with optimal tour edges as labels
                g, edge_features, node_features, labels = create_graph_from_instance(instance, optimal_tour)
                g = g.to(device)
                edge_features = edge_features.to(device)
                node_features = node_features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                edge_preds = model(g, edge_features, node_features)
                edge_preds = edge_preds.squeeze()
                
                # Compute loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(edge_preds, labels)
                batch_loss += loss
            
            # Average loss over batch
            batch_loss /= len(batch)
            total_loss += batch_loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create a unique run ID and directory for logs
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '_' + os.urandom(4).hex()
    log_dir = os.path.join(args.output_dir, 'logs', f'tspdl_gnn_{run_id}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Load datasets
    train_dataset = TSPDLDataset(
        os.path.join(args.data_dir, 'train'),
        device=device
    )
    val_dataset = TSPDLDataset(
        os.path.join(args.data_dir, 'val'),
        device=device
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch  # Identity collate function
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch  # Identity collate function
    )
    
    # Create model
    model = TSPDLEdgeModel(
        in_dim=3,  # [distance, load, draft_limit_diff]
        embed_dim=args.embed_dim,
        out_dim=1,
        n_layers=args.n_layers,
        n_heads=args.n_heads
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.n_epochs):
        # Train
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_time = time.time() - start_time
        
        # Validate
        start_time = time.time()
        val_loss = validate(model, val_loader, device)
        val_time = time.time() - start_time
        
        # Log
        print(f"Epoch {epoch+1}/{args.n_epochs}, "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved new best model with validation loss {val_loss:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to {args.output_dir}")
    print(f"Logs saved to {log_dir}")


if __name__ == '__main__':
    main()