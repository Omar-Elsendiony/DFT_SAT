"""
Complete Model Training Script with Critical Input Support
===========================================================

Expects data pre-split into fixed train/val/test directories.
Automatically resumes from an existing checkpoint if one exists.

Typical usage:
    # First time
    python train_gnn.py --train_dir data/train --val_dir data/val --test_dir data/test

    # After adding new .pkl files to data/train/ — just re-run the same command
    python train_gnn.py --train_dir data/train --val_dir data/val --test_dir data/test
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
        half_dim = out_dim // 2
        self.conv_fwd = GATv2Conv(in_dim, half_dim, heads=2, concat=False, dropout=dropout)
        self.conv_bwd = GATv2Conv(in_dim, half_dim, heads=2, concat=False, dropout=dropout)
        
    def forward(self, x, edge_index_fwd, edge_index_bwd):
        out_fwd = self.conv_fwd(x, edge_index_fwd)
        out_bwd = self.conv_bwd(x, edge_index_bwd)
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
        
        self.input_proj = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )
        
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(BiGNNLayer(hidden_dim, hidden_dim, dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        jk_dim = hidden_dim * (num_layers + 1)
        self.output_head = nn.Sequential(
            nn.Linear(jk_dim, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        src, dst = edge_index
        fwd_mask       = x[src, 8] < x[dst, 8]
        edge_index_fwd = edge_index[:, fwd_mask]
        edge_index_bwd = edge_index_fwd[[1, 0]]
        
        x  = self.input_proj(x)
        xs = [x]
        
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index_fwd, edge_index_bwd)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity
            xs.append(x)
        
        x_jk = torch.cat(xs, dim=1)
        return torch.sigmoid(self.output_head(x_jk))


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def weighted_bce_loss(pred, target, mask, importance_weights=None):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    if importance_weights is not None:
        bce = bce * importance_weights
    return (bce * mask).sum() / mask.sum().clamp(min=1)


def focal_loss(pred, target, mask, alpha=0.25, gamma=2.0):
    bce   = F.binary_cross_entropy(pred, target, reduction='none')
    p_t   = pred * target + (1 - pred) * (1 - target)
    focal = alpha * ((1 - p_t) ** gamma) * bce
    return (focal * mask).sum() / mask.sum().clamp(min=1)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, optimizer, device, use_focal=False, focal_alpha=0.25, focal_gamma=2.0):
    model.train()
    total_loss, num_batches = 0, 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred   = model(batch)
        mask   = batch.train_mask
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
    total_loss, total_correct, total_labeled, num_batches = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            pred   = model(batch)
            mask   = batch.train_mask
            target = batch.y_polarity
            
            total_loss    += weighted_bce_loss(pred, target, mask).item()
            pred_binary    = (pred > 0.5).float()
            total_correct += ((pred_binary == target) * mask).sum().item()
            total_labeled += mask.sum().item()
            num_batches   += 1
    
    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_labeled if total_labeled > 0 else 0.0
    return avg_loss, accuracy


# ============================================================================
# DATASET LOADING  (one directory at a time — RAM friendly)
# ============================================================================

def load_from_dir(data_dir, label):
    """Load all .pkl files from a single directory."""
    data      = []
    data_dir  = Path(data_dir)
    pkl_files = sorted(data_dir.glob('*.pkl'))

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in '{data_dir}'")

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data.extend(pickle.load(f))

    print(f"  {label}: {len(data)} samples from {len(pkl_files)} files  [{data_dir}]")
    return data


def load_dataset(train_dir, val_dir, test_dir):
    print("Loading datasets...")
    train_dataset = load_from_dir(train_dir, 'train')
    val_dataset   = load_from_dir(val_dir,   'val  ')
    test_dataset  = load_from_dir(test_dir,  'test ')

    # Only shuffle train — val and test stay in a consistent order
    np.random.shuffle(train_dataset)
    return train_dataset, val_dataset, test_dataset


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def try_load_checkpoint(save_path, model, device):
    """
    Always checks for an existing checkpoint at save_path on startup.
    Loads weights only — optimizer reinitializes fresh, which is correct
    when continuing with new data (old momentum terms would be stale).
    """
    if not os.path.exists(save_path):
        print("No existing checkpoint found. Starting fresh.")
        return 0, float('inf'), 0.0

    print(f"Found existing checkpoint at '{save_path}'. Loading weights to continue training...")
    checkpoint    = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    best_val_loss = checkpoint.get('val_loss',   float('inf'))
    best_val_acc  = checkpoint.get('val_acc',    0.0)
    prev_epoch    = checkpoint.get('epoch',      0)
    prev_data     = checkpoint.get('train_dir',  'unknown')

    print(f"  Previous run: epoch {prev_epoch + 1}, val_loss={best_val_loss:.4f}, "
          f"val_acc={best_val_acc:.4f}, train_dir='{prev_data}'")
    print("  Optimizer reinitialized fresh for new data.")
    return prev_epoch + 1, best_val_loss, best_val_acc


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset, val_dataset, test_dataset = load_dataset(
        args.train_dir, args.val_dir, args.test_dir
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)
    
    num_features = train_dataset[0].x.shape[1]
    print(f"Number of node features: {num_features}")
    
    model = CircuitGNN_Polarity(
        num_node_features=num_features,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")

    # Always check for an existing checkpoint before doing anything else
    start_epoch, best_val_loss, best_val_acc = try_load_checkpoint(args.save_path, model, device)

    # Use a reduced LR when continuing from a pretrained checkpoint
    effective_lr = args.lr if start_epoch == 0 else args.lr * 0.5
    if start_epoch > 0:
        print(f"Continuing training with reduced LR: {effective_lr} (was {args.lr})")

    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print("\nStarting training...")
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss        = train_epoch(model, train_loader, optimizer, device,
                                        args.use_focal, args.focal_alpha, args.focal_gamma)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_val_acc     = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             val_loss,
                'val_acc':              val_acc,
                'args':                 args,
                'train_dir':            args.train_dir,
            }, args.save_path)
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
    print(f"\nFinal Test Results:      Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")
    print(f"Best Validation Results: Loss={best_val_loss:.4f}, Accuracy={best_val_acc:.4f}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN for circuit polarity prediction')
    
    # Three fixed directories instead of one data_dir + internal split
    parser.add_argument('--train_dir',    type=str, required=True,  help='Directory with training .pkl files (add new data here)')
    parser.add_argument('--val_dir',      type=str, required=True,  help='Fixed validation directory — never changes between runs')
    parser.add_argument('--test_dir',     type=str, required=True,  help='Fixed test directory — never changes between runs')
    
    parser.add_argument('--num_layers',   type=int,   default=5,      help='Number of GNN layers')
    parser.add_argument('--hidden_dim',   type=int,   default=64,     help='Hidden dimension size')
    parser.add_argument('--dropout',      type=float, default=0.1,    help='Dropout rate')
    
    parser.add_argument('--epochs',       type=int,   default=8,      help='Maximum number of epochs per run')
    parser.add_argument('--batch_size',   type=int,   default=32,     help='Batch size')
    parser.add_argument('--lr',           type=float, default=0.001,  help='Learning rate (halved automatically when continuing)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,   help='Weight decay')
    parser.add_argument('--patience',     type=int,   default=30,     help='Early stopping patience')
    
    parser.add_argument('--use_focal',    action='store_true',        help='Use focal loss instead of BCE')
    parser.add_argument('--focal_alpha',  type=float, default=0.25,   help='Focal loss alpha')
    parser.add_argument('--focal_gamma',  type=float, default=2.0,    help='Focal loss gamma')
    
    parser.add_argument('--save_path',    type=str, default='best_model.pt', help='Path to save/load checkpoint')
    parser.add_argument('--seed',         type=int, default=42,              help='Random seed')
    
    args = parser.parse_args()
    
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model = train_model(args)
    print(f"\nTraining complete! Model saved to {args.save_path}")