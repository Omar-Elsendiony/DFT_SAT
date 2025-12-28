import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch_geometric.loader import DataLoader
from neuro_utils import CircuitGNN_Advanced

# --- CONFIG ---
DATASET_PATH = "dataset_oracle.pt"
MODEL_PATH = "gnn_model_oracle.pth"
EPOCHS = 20
BATCH_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_oracle():
    print("--- Training Oracle (Input Predictor) ---")
    
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found. Run generation script first!")
        return

    dataset = torch.load(DATASET_PATH, weights_only=False)
    
    split = int(len(dataset) * 0.8)
    train_data, val_data = dataset[:split], dataset[split:]
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 14 Features (Types + Depth + Fault + SCOAP)
    model = CircuitGNN_Advanced(num_node_features=14, num_layers=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loss: Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss(reduction='none') 
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch)
            
            # CRITICAL: Only learn from Primary Inputs
            raw_loss = criterion(preds, batch.y)
            masked_loss = (raw_loss * batch.train_mask).sum() / batch.train_mask.sum().clamp(min=1)
            
            masked_loss.backward()
            optimizer.step()
            total_loss += masked_loss.item()
            
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.5f}")
        
    torch.save(model.state_dict(), MODEL_PATH)
    print("--- Oracle Trained. Saved to gnn_model_oracle.pth ---")

if __name__ == "__main__":
    train_oracle()