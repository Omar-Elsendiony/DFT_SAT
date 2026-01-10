"""
Complete pipeline for importance-aware GNN training
This replaces your current generate_oracle_data.py and train_oracle.py
"""

import os
import sys
import time
import csv
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
# import torch.nn.functional as F
import torch.optim as optim
import random
from pysat.solvers import Glucose3
from pysat.formula import CNF
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from WireFaultMiter import WireFaultMiter
from neuro_utils import FastGraphExtractor


# Configs
BENCH_DIR = "../synthetic_bench"
DATASET_PATH = "dataset_oracle_importance.pt"
SAMPLES_PER_FILE = 50
MODEL_PATH = "gnn_model_importance_aware.pth"
EPOCHS = 20
BATCH_SIZE = 32


# =============================================================================
# PART 1: ENHANCED DATA GENERATION WITH IMPORTANCE
# =============================================================================

def get_target_files():
    if not os.path.exists(BENCH_DIR): return []
    return [f for f in os.listdir(BENCH_DIR) if f.endswith(".bench")]

def generate_importance_aware_dataset():
    """
    Enhanced version that collects:
    1. Correct solution (y_solution)
    2. Importance scores (y_importance) 
    3. Base conflict count (for analysis)
    """
    print(f"--- MINING IMPORTANCE-AWARE ORACLE DATA ---")
    dataset = []
    
    if not os.path.exists(BENCH_DIR):
        print(f"Error: {BENCH_DIR} not found.")
        return
    
    files = [f for f in os.listdir(BENCH_DIR) if f.endswith('.bench')]
    
    for filename in tqdm(files, desc="Mining Circuits"):
        filepath = os.path.join(BENCH_DIR, filename)
        
        try:
            miter = WireFaultMiter(filepath)
            if not miter.gates:
                continue
            
            extractor = FastGraphExtractor(filepath, miter.var_map)
            input_set = set(miter.inputs)
            
            for _ in range(SAMPLES_PER_FILE):
                target_gate = random.choice(miter.gates)[0]
                
                # === STEP 1: Get correct solution ===
                clauses = miter.build_miter(target_gate, None, 1)
                cnf = CNF()
                cnf.extend(clauses)
                
                with Glucose3(bootstrap_with=cnf) as solver:
                    if not solver.solve():
                        continue  # Skip UNSAT cases
                    
                    model = solver.get_model()
                    if not model:
                        continue
                    
                    base_conflicts = solver.accum_stats()['conflicts']
                    
                    # === STEP 2: Measure importance per input ===
                    input_importance = {}
                    
                    for input_name in miter.inputs:
                        var_id = miter.var_map[input_name]
                        
                        # Get the correct value from the model
                        correct_val = var_id if var_id in model else -var_id
                        wrong_val = -correct_val
                        
                        # Test: What happens if we force WRONG value?
                        test_cnf = CNF()
                        test_cnf.extend(clauses)
                        
                        with Glucose3(bootstrap_with=test_cnf) as test_solver:
                            # Force wrong assignment
                            result = test_solver.solve(assumptions=[wrong_val])
                            
                            if result:  
                                # Still SAT with wrong value -> measure extra work
                                wrong_conflicts = test_solver.accum_stats()['conflicts']
                                importance = abs(wrong_conflicts - base_conflicts)
                            else:  
                                # UNSAT with wrong value -> this input is CRITICAL
                                importance = 10000
                        
                        input_importance[input_name] = importance
                    
                    # Normalize importance scores to 0-1 range
                    max_importance = max(input_importance.values()) if input_importance else 1
                    normalized_importance = {
                        k: v / max(max_importance, 1.0) 
                        for k, v in input_importance.items()
                    }
                    
                    # === STEP 3: Create training data ===
                    data = extractor.get_data_for_fault(target_gate)
                    
                    # Prepare label tensors
                    y_solution = torch.zeros(len(data.node_names), 1)
                    y_importance = torch.zeros(len(data.node_names), 1)
                    train_mask = torch.zeros(len(data.node_names), 1)
                    
                    # Fill in labels for input nodes only
                    for i, node_name in enumerate(data.node_names):
                        if node_name in input_set:
                            var_id = miter.var_map[node_name]
                            
                            # Label 1: What value should this input have?
                            y_solution[i] = 1.0 if var_id in model else 0.0
                            
                            # Label 2: How important is getting this right?
                            y_importance[i] = normalized_importance.get(node_name, 0.0)
                            
                            # Mask: Learn from this node
                            train_mask[i] = 1.0
                    
                    # Attach all labels to the data object
                    data.y = y_solution              # What value (0 or 1)
                    data.y_importance = y_importance # How important (0 to 1)
                    data.train_mask = train_mask     # Which nodes to learn from
                    data.base_conflicts = base_conflicts  # For analysis/debugging
                    
                    dataset.append(data)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"--- Mining Complete. Collected {len(dataset)} samples. ---")
    torch.save(dataset, DATASET_PATH)
    print(f"--- Dataset saved to {DATASET_PATH} ---")
    return dataset


# =============================================================================
# PART 2: MULTI-HEAD GNN MODEL (Predicts BOTH value AND importance)
# =============================================================================

class CircuitGNN_ImportanceAware(torch.nn.Module):
    """
    Enhanced GNN with two prediction heads:
    1. Value head: Should this input be 0 or 1?
    2. Importance head: How critical is this assignment?
    """
    
    def __init__(self, num_node_features=14, num_layers=8, hidden_dim=64, dropout=0.2):
        super(CircuitGNN_ImportanceAware, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Input Layer
        self.convs.append(GATv2Conv(num_node_features, hidden_dim, heads=2, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden Layers (Residuals)
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output Layer (shared representation)
        self.convs.append(GATv2Conv(hidden_dim, 32, heads=2, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(32))
        
        # TWO prediction heads
        self.value_head = torch.nn.Linear(32, 1)       # Predicts 0 or 1
        self.importance_head = torch.nn.Linear(32, 1)  # Predicts importance score
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Shared GNN backbone
        x = self.convs[0](x, edge_index)
        x = self.bns[0](x)
        x = torch.nn.functional.elu(x)
        
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = torch.nn.functional.elu(x)
            x = x + identity  # Residual
        
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = torch.nn.functional.elu(x)
        
        # Two separate predictions
        value_logits = self.value_head(x)           # For BCE loss
        importance_scores = self.importance_head(x)  # For MSE loss
        
        return value_logits, importance_scores


# =============================================================================
# PART 3: TRAINING WITH MULTI-TASK LOSS
# =============================================================================

def train_importance_aware_model():
    """
    Train model with two objectives:
    1. Predict correct input values (BCE loss)
    2. Predict input importance (MSE loss)
    """
    print("--- Training Importance-Aware Oracle ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        print("Generating dataset first...")
        generate_importance_aware_dataset()
    
    # Load dataset
    dataset = torch.load(DATASET_PATH, weights_only=False)
    print(f"Loaded {len(dataset)} samples")
    
    # Train/Val split
    split = int(len(dataset) * 0.8)
    train_data, val_data = dataset[:split], dataset[split:]
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = CircuitGNN_ImportanceAware(
        num_node_features=14, 
        num_layers=8, 
        dropout=0.2
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loss functions
    value_criterion = nn.BCEWithLogitsLoss(reduction='none')
    importance_criterion = nn.MSELoss(reduction='none')
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_value_loss = 0
        total_importance_loss = 0
        total_combined_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass (two outputs)
            value_logits, importance_preds = model(batch)
            
            # Loss 1: Value prediction (BCE)
            value_loss_raw = value_criterion(value_logits, batch.y)
            value_loss_masked = (value_loss_raw * batch.train_mask).sum() / batch.train_mask.sum().clamp(min=1)
            
            # Loss 2: Importance prediction (MSE)
            importance_loss_raw = importance_criterion(importance_preds, batch.y_importance)
            importance_loss_masked = (importance_loss_raw * batch.train_mask).sum() / batch.train_mask.sum().clamp(min=1)
            
            # Combined loss (weighted)
            # Value is more important (60%) than importance ranking (40%)
            combined_loss = 0.6 * value_loss_masked + 0.4 * importance_loss_masked
            
            # Backprop
            combined_loss.backward()
            optimizer.step()
            
            total_value_loss += value_loss_masked.item()
            total_importance_loss += importance_loss_masked.item()
            total_combined_loss += combined_loss.item()
        
        # Print stats
        avg_value = total_value_loss / len(train_loader)
        avg_importance = total_importance_loss / len(train_loader)
        avg_combined = total_combined_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Value Loss: {avg_value:.5f}")
        print(f"  Importance Loss: {avg_importance:.5f}")
        print(f"  Combined Loss: {avg_combined:.5f}")
    
    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n--- Training Complete. Model saved to {MODEL_PATH} ---")


# =============================================================================
# PART 4: INFERENCE (Using importance for ranking)
# =============================================================================

def run_importance_guided_benchmark():
    """
    Use the trained model to guide SAT solving.
    Now uses BOTH value predictions AND importance scores.
    """

    print(f"--- BENCHMARKING WITH IMPORTANCE-GUIDED HINTS ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = CircuitGNN_ImportanceAware(num_node_features=14, num_layers=8).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Train first!")
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    results = []
    files = get_target_files()
    
    for filename in files:
        filepath = os.path.join(BENCH_DIR, filename)
        print(f"\nProcessing {filename}...")
        
        try:
            miter = WireFaultMiter(filepath)
            if not miter.gates:
                continue
            
            extractor = FastGraphExtractor(filepath, miter.var_map)
            input_names = set(miter.inputs)
            
            for i in range(20):
                target_gate = random.choice(miter.gates)[0]
                
                # Baseline SAT
                clauses = miter.build_miter(target_gate, None, 1)
                cnf = CNF()
                cnf.extend(clauses)
                
                t_std = time.time()
                with Glucose3(bootstrap_with=cnf) as solver:
                    solver.solve()
                    std_conflicts = solver.accum_stats()['conflicts']
                std_time = time.time() - t_std
                
                # GNN-Guided SAT
                t_gnn = time.time()
                data = extractor.get_data_for_fault(target_gate)
                if data is None:
                    continue
                data = data.to(device)
                
                with torch.no_grad():
                    value_logits, importance_scores = model(data)
                    value_probs = torch.sigmoid(value_logits)  # 0-1 probabilities
                
                # Extract hints with importance ranking
                hints = []
                for idx, name in enumerate(data.node_names):
                    if name in input_names:
                        # Value prediction
                        prob = value_probs[idx].item()
                        value = 1 if prob > 0.5 else -1
                        
                        # Importance score (how critical is this variable?)
                        importance = importance_scores[idx].item()
                        
                        var_id = miter.var_map.get(name)
                        if var_id:
                            hints.append((var_id, value, importance))
                
                # Sort by IMPORTANCE (not value confidence)
                # This is the key difference: we prioritize based on importance!
                hints.sort(key=lambda x: x[2], reverse=True)
                
                # Take top K most important inputs
                TOP_K = 5
                top_assumptions = [h[0] * h[1] for h in hints[:TOP_K]]
                
                # Solve with importance-ranked assumptions
                gnn_conflicts = 0
                with Glucose3(bootstrap_with=cnf) as solver:
                    if solver.solve(assumptions=top_assumptions):
                        gnn_conflicts = solver.accum_stats()['conflicts']
                    else:
                        solver.solve()
                        gnn_conflicts = solver.accum_stats()['conflicts'] + 1000
                
                gnn_time = time.time() - t_gnn
                
                # Report
                speedup = std_conflicts / max(gnn_conflicts, 1)
                time_speedup = std_time / max(gnn_time, 0.0001)
                
                print(f"   Fault: {target_gate}")
                print(f"     Conflicts: {std_conflicts} -> {gnn_conflicts} ({speedup:.2f}x)")
                print(f"     Time: {std_time:.4f}s -> {gnn_time:.4f}s")
                
                results.append({
                    "Circuit": filename,
                    "Fault": target_gate,
                    "Std_Conflicts": std_conflicts,
                    "GNN_Conflicts": gnn_conflicts,
                    "Speedup": f"{speedup:.2f}x",
                    "Time_Speedup": f"{time_speedup:.2f}x"
                })
        
        except Exception as e:
            print(f"Error: {e}")
    
    # Save results
    if results:
        with open("importance_guided_results.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to importance_guided_results.csv")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python importance_pipeline.py generate  # Generate dataset")
        print("  python importance_pipeline.py train     # Train model")
        print("  python importance_pipeline.py benchmark # Run benchmark")
    else:
        command = sys.argv[1]
        
        if command == "generate":
            generate_importance_aware_dataset()
        elif command == "train":
            train_importance_aware_model()
        elif command == "benchmark":
            run_importance_guided_benchmark()
        else:
            print(f"Unknown command: {command}")