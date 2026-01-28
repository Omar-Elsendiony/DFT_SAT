"""
GNN Training Script for Dual-Task Learning (Polarity + Importance)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import DataLoader
import random
import numpy as np

# =============================================================================
# CONFIGS
# =============================================================================
DATASET_PATH = "dataset_complete_atpg_17feat.pt"
MODEL_PATH = "gnn_model_dual_task_17feat.pth"
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 42

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_global_seed(SEED)

# =============================================================================
# MODEL DEFINITION
# =============================================================================

class CircuitGNN_DualTask(torch.nn.Module):
    """
    Dual-task GNN for circuit testability prediction.
    
    Predicts:
    1. Input Importance (regression)
    2. Input Polarity (binary classification)
    """
    def __init__(self, num_node_features=17, num_layers=20, hidden_dim=64, dropout=0.2):
        super(CircuitGNN_DualTask, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GATv2Conv(num_node_features, hidden_dim, heads=2, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers with residual connections
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GATv2Conv(hidden_dim, 32, heads=2, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(32))
        
        # Task-specific heads
        self.importance_head = torch.nn.Linear(32, 1)
        self.polarity_head = torch.nn.Linear(32, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First layer
        x = self.convs[0](x, edge_index)
        x = self.bns[0](x)
        x = torch.nn.functional.elu(x)
        
        # Middle layers with residual connections
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = torch.nn.functional.elu(x)
            x = x + identity  # Residual connection
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = torch.nn.functional.elu(x)
        
        # Task heads
        importance = self.importance_head(x)
        polarity = torch.sigmoid(self.polarity_head(x))
        
        return importance, polarity

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model():
    print("=" * 80)
    print("DUAL-TASK GNN TRAINING")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("Please run data generation first!")
        return
    
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = torch.load(DATASET_PATH, weights_only=False)
    print(f"Loaded {len(dataset)} samples")
    
    # Split dataset
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    model = CircuitGNN_DualTask(num_node_features=17).to(device)
    
    # Check if pre-trained model exists
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Training from scratch...")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss functions
    criterion_importance = nn.MSELoss(reduction='none')
    criterion_polarity = nn.BCELoss(reduction='none')
    
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_imp_loss = 0.0
        train_pol_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            pred_importance, pred_polarity = model(batch)
            
            # Compute masked losses
            mask = batch.train_mask
            mask_sum = mask.sum().clamp(min=1)
            
            loss_imp = (criterion_importance(pred_importance, batch.y_importance) * mask).sum() / mask_sum
            loss_pol = (criterion_polarity(pred_polarity, batch.y_polarity) * mask).sum() / mask_sum
            
            total_loss = loss_imp + loss_pol
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_imp_loss += loss_imp.item()
            train_pol_loss += loss_pol.item()
        
        train_loss /= len(train_loader)
        train_imp_loss /= len(train_loader)
        train_pol_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_imp_loss = 0.0
        val_pol_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_importance, pred_polarity = model(batch)
                
                mask = batch.train_mask
                mask_sum = mask.sum().clamp(min=1)
                
                loss_imp = (criterion_importance(pred_importance, batch.y_importance) * mask).sum() / mask_sum
                loss_pol = (criterion_polarity(pred_polarity, batch.y_polarity) * mask).sum() / mask_sum
                
                val_loss += (loss_imp + loss_pol).item()
                val_imp_loss += loss_imp.item()
                val_pol_loss += loss_pol.item()
        
        val_loss /= len(val_loader)
        val_imp_loss /= len(val_loader)
        val_pol_loss /= len(val_loader)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} (Imp: {train_imp_loss:.4f}, Pol: {train_pol_loss:.4f}) | "
              f"Val Loss: {val_loss:.4f} (Imp: {val_imp_loss:.4f}, Pol: {val_pol_loss:.4f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()