"""
Complete Model Training Script with Critical Input Support
===========================================================

This training script is optimized for the critical input filtered dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import os
import pickle
from pathlib import Path
import argparse


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class BiGNNLayer(nn.Module):
    """Bidirectional message passing layer to separate input/output logic."""
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        # Ensure out_dim is divisible by 2 for concatenation
        half_dim = out_dim // 2
        
        # Forward edges: what outputs hear from inputs
        self.conv_fwd = GATv2Conv(in_dim, half_dim, heads=2, concat=False, dropout=dropout)
        
        # Backward edges: what inputs hear from outputs
        self.conv_bwd = GATv2Conv(in_dim, half_dim, heads=2, concat=False, dropout=dropout)
        
    def forward(self, x, edge_index_fwd, edge_index_bwd):
        out_fwd = self.conv_fwd(x, edge_index_fwd)
        out_bwd = self.conv_bwd(x, edge_index_bwd)
        # Combine forward and backward message passing
        return torch.cat([out_fwd, out_bwd], dim=-1)


class CircuitGNN_Polarity(torch.nn.Module):
    """
    GNN for predicting input polarities in circuit ATPG.
    
    Optimized with:
    - Bi-directional message passing (distinguishes upstream vs downstream)
    - Shallower architecture (5 layers) to prevent oversmoothing
    - Jumping Knowledge (concatenating all layers)
    """
    
    def __init__(self, num_node_features=17, num_layers=5, hidden_dim=64, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )
        
        # Bi-directional GNN layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(BiGNNLayer(hidden_dim, hidden_dim, dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Output head with Jumping Knowledge 
        # (hidden_dim for original proj + hidden_dim for each layer)
        jk_dim = hidden_dim * (num_layers + 1)
        self.output_head = nn.Sequential(
            nn.Linear(jk_dim, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Dynamically separate forward and backward edges
        # Feature index 8 is topological forward depth. Forward edges always flow to a higher depth.
        src, dst = edge_index
        fwd_mask = x[src, 8] < x[dst, 8]
        edge_index_fwd = edge_index[:, fwd_mask]
        edge_index_bwd = edge_index_fwd[[1, 0]]  # The exact reverse of forward edges
        
        # Input projection
        x = self.input_proj(x)
        
        # Store layer outputs for Jumping Knowledge
        xs = [x]
        
        # GNN layers with residual connections
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index_fwd, edge_index_bwd)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity  # Residual connection
            xs.append(x)
        
        # Concatenate original projection + all GAT layers
        x_jk = torch.cat(xs, dim=1) 
        
        # Output
        x_out = self.output_head(x_jk)
        return torch.sigmoid(x_out)


# ============================================================================
# IMPROVED LOSS FUNCTION
# ============================================================================

def weighted_bce_loss(pred, target, mask, importance_weights=None):
    """Binary cross-entropy loss with optional importance weighting."""
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    if importance_weights is not None:
        bce = bce * importance_weights
    masked_loss = (bce * mask).sum() / mask.sum().clamp(min=1)
    return masked_loss


def focal_loss(pred, target, mask, alpha=0.25, gamma=2.0):
    """Focal loss to handle class imbalance in critical inputs."""
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    p_t = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    focal = alpha * focal_weight * bce
    masked_loss = (focal * mask).sum() / mask.sum().clamp(min=1)
    return masked_loss


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_epoch(model, loader, optimizer, device, use_focal=False, focal_alpha=0.25, focal_gamma=2.0):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch)
        mask = batch.train_mask
        target = batch.y_polarity
        
        if use_focal:
            loss = focal_loss(pred, target, mask, focal_alpha, focal_gamma)
        else:
            importance = batch.y_importance if hasattr(batch, 'y_importance') else None
            loss = weighted_bce_loss(pred, target, mask, importance)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_labeled = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            mask = batch.train_mask
            target = batch.y_polarity
            
            loss = weighted_bce_loss(pred, target, mask)
            total_loss += loss.item()
            
            pred_binary = (pred > 0.5).float()
            correct = ((pred_binary == target) * mask).sum().item()
            labeled = mask.sum().item()
            
            total_correct += correct
            total_labeled += labeled
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_labeled if total_labeled > 0 else 0.0
    
    return avg_loss, accuracy


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(data_dir, train_ratio=0.8, val_ratio=0.1):
    all_data = []
    data_dir = Path(data_dir)
    for pkl_file in data_dir.glob('*.pkl'):
        print(f"Loading {pkl_file.name}...")
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    
    print(f"Loaded {len(all_data)} samples total")
    np.random.shuffle(all_data)
    
    n = len(all_data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_dataset = all_data[:n_train]
    val_dataset = all_data[n_train:n_train + n_val]
    test_dataset = all_data[n_train + n_val:]
    
    print(f"Split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    return train_dataset, val_dataset, test_dataset


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    train_dataset, val_dataset, test_dataset = load_dataset(args.data_dir, args.train_ratio, args.val_ratio)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    num_features = train_dataset[0].x.shape[1]
    print(f"Number of node features: {num_features}")
    
    print("Creating model...")
    model = CircuitGNN_Polarity(
        num_node_features=num_features,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.use_focal, args.focal_alpha, args.focal_gamma)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': args
            }
            torch.save(checkpoint, args.save_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(args.save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    print(f"Best Validation Results (epoch {checkpoint['epoch']+1}): Loss: {best_val_loss:.4f}, Accuracy: {best_val_acc:.4f}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN for circuit polarity prediction')
    
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data (.pkl files)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Fraction of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Fraction of data for validation')
    
    parser.add_argument('--num_layers', type=int, default=5, help='Number of GNN layers')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    parser.add_argument('--epochs', type=int, default=8, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    
    parser.add_argument('--use_focal', action='store_true', help='Use focal loss instead of BCE')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter')
    
    parser.add_argument('--save_path', type=str, default='best_model.pt', help='Path to save best model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model = train_model(args)
    print(f"\nTraining complete! Model saved to {args.save_path}")