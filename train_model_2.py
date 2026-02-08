import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import DataLoader

# =============================================================================
# CONFIGS
# =============================================================================
DATASET_PATH = "dataset_complete_atpg_17feat.pt"
MODEL_PATH = "gnn_model_polarity_only.pth"
EPOCHS = 8           # Increased slightly as training is now faster/lighter
BATCH_SIZE = 32        # Kept low to prevent CUDA OOM
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5 
SEED = 42

# Architecture
NUM_LAYERS = 8        # Deeper than 10-12 often causes over-smoothing in circuits
HIDDEN_DIM = 32       # Lowered to 32 to save VRAM

# Training hyperparameters
PATIENCE = 10
GRADIENT_CLIP = 1.0

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
# MODEL DEFINITION (POLARITY ONLY)
# =============================================================================

class CircuitGNN_Polarity(torch.nn.Module):
    """GNN focused exclusively on predicting signal polarity for ATPG"""
    def __init__(self, num_node_features=17, num_layers=10, hidden_dim=32, dropout=0.1):
        super(CircuitGNN_Polarity, self).__init__()
        self.num_layers = num_layers
        self.dropout_layer = torch.nn.Dropout(dropout)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Input layer: Mapping 17 features to hidden_dim
        self.convs.append(GATv2Conv(num_node_features, hidden_dim, heads=2, concat=False, dropout=dropout))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers with Residual-ready structure
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output compression layer
        self.convs.append(GATv2Conv(hidden_dim, 16, heads=2, concat=False, dropout=dropout))
        self.bns.append(torch.nn.BatchNorm1d(16))
        
        self.polarity_head = torch.nn.Linear(16, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = torch.nn.functional.elu(x)
            x = self.dropout_layer(x)
            
            # Apply residual connection if shapes match
            if i > 0 and x.shape == identity.shape:
                x = x + identity
        
        return torch.sigmoid(self.polarity_head(x))

# =============================================================================
# METRICS & TRAINING
# =============================================================================

def compute_accuracy(pred_polarity, y_polarity, mask):
    with torch.no_grad():
        pred_binary = (pred_polarity > 0.5).float()
        correct = ((pred_binary == y_polarity).float() * mask).sum()
        total = mask.sum().clamp(min=1)
        return (correct / total).item()

def train_model():
    # Force memory cleanup before starting
    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 80)
    print("GNN POLARITY-ONLY TRAINING")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        return
    
    dataset = torch.load(DATASET_PATH, weights_only=False)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # Split 70/15/15
    train_end, val_end = int(0.7 * len(indices)), int(0.85 * len(indices))
    train_loader = DataLoader([dataset[i] for i in indices[:train_end]], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in indices[train_end:val_end]], batch_size=BATCH_SIZE)
    test_loader = DataLoader([dataset[i] for i in indices[val_end:]], batch_size=BATCH_SIZE)
    
    model = CircuitGNN_Polarity(num_node_features=17, num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.BCELoss(reduction='none')

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss, total_train_acc = 0, 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch)
            mask = batch.train_mask
            loss = (criterion(pred, batch.y_polarity) * mask).sum() / mask.sum().clamp(min=1)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_acc += compute_accuracy(pred, batch.y_polarity, mask)

        # Validation
        model.eval()
        total_val_loss, total_val_acc = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                mask = batch.train_mask
                v_loss = (criterion(pred, batch.y_polarity) * mask).sum() / mask.sum().clamp(min=1)
                total_val_loss += v_loss.item()
                total_val_acc += compute_accuracy(pred, batch.y_polarity, mask)

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:02d} | Train Loss: {total_train_loss/len(train_loader):.4f} Acc: {total_train_acc/len(train_loader):.3f} | Val Loss: {avg_val_loss:.4f} Acc: {total_val_acc/len(val_loader):.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Final Test
    print("\n--- Final Test Evaluation ---")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    test_acc = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            test_acc += compute_accuracy(pred, batch.y_polarity, batch.train_mask)
    print(f"Final Test Accuracy: {test_acc/len(test_loader):.3f}")

if __name__ == "__main__":
    train_model()