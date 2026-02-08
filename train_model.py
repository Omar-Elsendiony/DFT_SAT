"""
Enhanced Training Script with Best Practices for GNN-Guided ATPG

IMPROVEMENTS:
1. Train/Val/Test split (70/15/15) with shuffle
2. Learning rate scheduling (ReduceLROnPlateau)
3. Gradient clipping (prevents exploding gradients)
4. Dropout regularization (prevents overfitting)
5. Weight decay (L2 regularization)
6. Early stopping (patience=15)
7. Accuracy metrics (not just loss)
8. Final test evaluation
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

import torch
import gc

# Add this at the very top of your train_model() function
gc.collect()
torch.cuda.empty_cache()
# =============================================================================
# CONFIGS
# =============================================================================
DATASET_PATH = "dataset_complete_atpg_17feat.pt"
MODEL_PATH = "gnn_model_dual_task_17feat_improved.pth"
EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5  # L2 regularization
SEED = 42

# Training hyperparameters
PATIENCE = 15
GRADIENT_CLIP = 1.0
TASK_WEIGHTS = {'importance': 0.5, 'polarity': 0.5}

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
# MODEL DEFINITION (ENHANCED WITH DROPOUT)
# =============================================================================

class CircuitGNN_DualTask(torch.nn.Module):
    """Enhanced GNN with dropout for regularization"""
    def __init__(self, num_node_features=17, num_layers=10, hidden_dim=64, dropout=0.1):
        super(CircuitGNN_DualTask, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GATv2Conv(num_node_features, hidden_dim, heads=2, concat=False, dropout=dropout))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GATv2Conv(hidden_dim, 32, heads=2, concat=False, dropout=dropout))
        self.bns.append(torch.nn.BatchNorm1d(32))
        
        # Task heads with dropout
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.importance_head = torch.nn.Linear(32, 1)
        self.polarity_head = torch.nn.Linear(32, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First layer
        x = self.convs[0](x, edge_index)
        x = self.bns[0](x)
        x = torch.nn.functional.elu(x)
        x = self.dropout_layer(x)
        
        # Middle layers with residual
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = torch.nn.functional.elu(x)
            x = self.dropout_layer(x)
            x = x + identity
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = torch.nn.functional.elu(x)
        x = self.dropout_layer(x)
        
        # Prediction heads
        importance = self.importance_head(x)
        polarity = torch.sigmoid(self.polarity_head(x))
        
        return importance, polarity

# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(pred_polarity, y_polarity, train_mask):
    """Compute accuracy for polarity prediction"""
    with torch.no_grad():
        pred_binary = (pred_polarity > 0.5).float()
        correct = ((pred_binary == y_polarity).float() * train_mask).sum()
        total = train_mask.sum()
        accuracy = (correct / total).item() if total > 0 else 0.0
    return accuracy

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model():
    print("=" * 80)
    print("ENHANCED DUAL-TASK GNN TRAINING")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        return
    
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = torch.load(DATASET_PATH, weights_only=False)
    print(f"Loaded {len(dataset)} samples")
    
    # Shuffle and split: 70% train, 15% val, 15% test
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    train_end = int(0.7 * len(indices))
    val_end = int(0.85 * len(indices))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = CircuitGNN_DualTask(num_node_features=17, dropout=0.1).to(device)
    
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss functions
    criterion_importance = nn.MSELoss(reduction='none')
    criterion_polarity = nn.BCELoss(reduction='none')
    
    print("\n" + "=" * 80)
    print("Training Started")
    print("=" * 80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # ==================== TRAINING ====================
        model.train()
        train_loss = 0.0
        train_imp_loss = 0.0
        train_pol_loss = 0.0
        train_accuracy = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred_importance, pred_polarity = model(batch)
            
            mask = batch.train_mask
            mask_sum = mask.sum().clamp(min=1)
            
            loss_imp = (criterion_importance(pred_importance, batch.y_importance) * mask).sum() / mask_sum
            loss_pol = (criterion_polarity(pred_polarity, batch.y_polarity) * mask).sum() / mask_sum
            
            total_loss = TASK_WEIGHTS['importance'] * loss_imp + TASK_WEIGHTS['polarity'] * loss_pol
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            train_imp_loss += loss_imp.item()
            train_pol_loss += loss_pol.item()
            train_accuracy += compute_metrics(pred_polarity, batch.y_polarity, mask)
        
        train_loss /= len(train_loader)
        train_imp_loss /= len(train_loader)
        train_pol_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        
        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0.0
        val_imp_loss = 0.0
        val_pol_loss = 0.0
        val_accuracy = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_importance, pred_polarity = model(batch)
                
                mask = batch.train_mask
                mask_sum = mask.sum().clamp(min=1)
                
                loss_imp = (criterion_importance(pred_importance, batch.y_importance) * mask).sum() / mask_sum
                loss_pol = (criterion_polarity(pred_polarity, batch.y_polarity) * mask).sum() / mask_sum
                
                total_loss = TASK_WEIGHTS['importance'] * loss_imp + TASK_WEIGHTS['polarity'] * loss_pol
                
                val_loss += total_loss.item()
                val_imp_loss += loss_imp.item()
                val_pol_loss += loss_pol.item()
                val_accuracy += compute_metrics(pred_polarity, batch.y_polarity, mask)
        
        val_loss /= len(val_loader)
        val_imp_loss /= len(val_loader)
        val_pol_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | LR: {current_lr:.6f}")
        print(f"  Train - Loss: {train_loss:.4f} | Imp: {train_imp_loss:.4f} | Pol: {train_pol_loss:.4f} | Acc: {train_accuracy:.3f}")
        print(f"  Val   - Loss: {val_loss:.4f} | Imp: {val_imp_loss:.4f} | Pol: {val_pol_loss:.4f} | Acc: {val_accuracy:.3f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
    
    # ==================== TEST EVALUATION ====================
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    test_loss = 0.0
    test_imp_loss = 0.0
    test_pol_loss = 0.0
    test_accuracy = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_importance, pred_polarity = model(batch)
            
            mask = batch.train_mask
            mask_sum = mask.sum().clamp(min=1)
            
            loss_imp = (criterion_importance(pred_importance, batch.y_importance) * mask).sum() / mask_sum
            loss_pol = (criterion_polarity(pred_polarity, batch.y_polarity) * mask).sum() / mask_sum
            
            test_loss += (TASK_WEIGHTS['importance'] * loss_imp + TASK_WEIGHTS['polarity'] * loss_pol).item()
            test_imp_loss += loss_imp.item()
            test_pol_loss += loss_pol.item()
            test_accuracy += compute_metrics(pred_polarity, batch.y_polarity, mask)
    
    test_loss /= len(test_loader)
    test_imp_loss /= len(test_loader)
    test_pol_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"  Importance Loss: {test_imp_loss:.4f}")
    print(f"  Polarity Loss: {test_pol_loss:.4f}")
    print(f"  Polarity Accuracy: {test_accuracy:.3f}")
    print("=" * 80)
    print(f"Best model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()