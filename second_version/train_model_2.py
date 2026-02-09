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
from datetime import datetime


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class CircuitGNN_Polarity(torch.nn.Module):
    """
    GNN for predicting input polarities in circuit ATPG.
    
    Optimized for critical input learning with:
    - Deeper architecture (12 layers)
    - Residual connections
    - Better normalization
    """
    
    def __init__(self, num_node_features=17, num_layers=12, hidden_dim=64, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )
        
        # GNN layers with residual connections
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=dropout)
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        
        # GNN layers with residual connections
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity  # Residual connection
        
        # Output
        x = self.output_head(x)
        return torch.sigmoid(x)


# ============================================================================
# IMPROVED LOSS FUNCTION
# ============================================================================

def weighted_bce_loss(pred, target, mask, importance_weights=None):
    """
    Binary cross-entropy loss with optional importance weighting.
    
    Args:
        pred: Predicted polarities [N, 1]
        target: Ground truth polarities [N, 1]
        mask: Training mask [N, 1] (1.0 for labeled nodes, 0.0 otherwise)
        importance_weights: Optional [N, 1] weights for each node
    
    Returns:
        Weighted loss scalar
    """
    # Compute BCE for all nodes
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    
    # Apply importance weights if provided
    if importance_weights is not None:
        bce = bce * importance_weights
    
    # Apply mask and average over labeled nodes
    masked_loss = (bce * mask).sum() / mask.sum().clamp(min=1)
    
    return masked_loss


def focal_loss(pred, target, mask, alpha=0.25, gamma=2.0):
    """
    Focal loss to handle class imbalance in critical inputs.
    
    Focuses training on hard examples (inputs with uncertain predictions).
    """
    # Compute BCE
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    
    # Compute focal weight
    p_t = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    
    # Combine
    focal = alpha * focal_weight * bce
    
    # Apply mask
    masked_loss = (focal * mask).sum() / mask.sum().clamp(min=1)
    
    return masked_loss


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_epoch(model, loader, optimizer, device, use_focal=False, focal_alpha=0.25, focal_gamma=2.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch)
        
        # Get mask and targets
        mask = batch.train_mask
        target = batch.y_polarity
        
        # Compute loss
        if use_focal:
            loss = focal_loss(pred, target, mask, focal_alpha, focal_gamma)
        else:
            # Use importance weights if available
            importance = batch.y_importance if hasattr(batch, 'y_importance') else None
            loss = weighted_bce_loss(pred, target, mask, importance)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, loader, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_labeled = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass
            pred = model(batch)
            
            # Get mask and targets
            mask = batch.train_mask
            target = batch.y_polarity
            
            # Compute loss
            loss = weighted_bce_loss(pred, target, mask)
            total_loss += loss.item()
            
            # Compute accuracy on labeled nodes
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
    """
    Load and split dataset.
    
    Args:
        data_dir: Directory containing .pkl files
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (rest is test)
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    all_data = []
    
    # Load all pickle files
    data_dir = Path(data_dir)
    for pkl_file in data_dir.glob('*.pkl'):
        print(f"Loading {pkl_file.name}...")
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    
    print(f"Loaded {len(all_data)} samples total")
    
    # Shuffle
    np.random.shuffle(all_data)
    
    # Split
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
    """Main training function."""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    train_dataset, val_dataset, test_dataset = load_dataset(
        args.data_dir, args.train_ratio, args.val_ratio
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get number of node features from first sample
    num_features = train_dataset[0].x.shape[1]
    print(f"Number of node features: {num_features}")
    
    # Create model
    print("Creating model...")
    model = CircuitGNN_Polarity(
        num_node_features=num_features,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            use_focal=args.use_focal,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, "
              f"val_acc={val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
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
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate on test set
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(args.save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"\nBest Validation Results (epoch {checkpoint['epoch']+1}):")
    print(f"  Loss: {best_val_loss:.4f}")
    print(f"  Accuracy: {best_val_acc:.4f}")
    
    return model


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN for circuit polarity prediction')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data (.pkl files)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Fraction of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Fraction of data for validation')
    
    # Model
    parser.add_argument('--num_layers', type=int, default=12,
                       help='Number of GNN layers')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension size')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3,
                       help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience')
    
    # Loss function
    parser.add_argument('--use_focal', action='store_true',
                       help='Use focal loss instead of BCE')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')
    
    # Misc
    parser.add_argument('--save_path', type=str, default='best_model.pt',
                       help='Path to save best model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create save directory if needed
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Train
    model = train_model(args)
    
    print(f"\nTraining complete! Model saved to {args.save_path}")