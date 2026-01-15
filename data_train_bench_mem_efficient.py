"""
Complete pipeline for Dual-Task GNN training with Parallel Generation & Polarity Guidance.
Uses 'set_phases' for soft solver constraints.
"""

import os
import sys

# Add local PySAT if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pysat'))

import time
import csv
import math
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import torch.optim as optim
import random
from pysat.solvers import Glucose3
from pysat.formula import CNF
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from WireFaultMiter import WireFaultMiter
from BenchParser import BenchParser
from torch_geometric.data import Data
import torch.multiprocessing as mp

# IMPORT THE EXTRACTOR
from neuro_utils import VectorizedGraphExtractor

# =============================================================================
# CONFIGS
# =============================================================================
BENCH_DIR = "../hdl-benchmarks/iscas85/bench/"
DATASET_PATH = "dataset_oracle_dual_16feat.pt"
SAMPLES_PER_FILE = 50
MODEL_PATH = "gnn_model_dual_task_16feat.pth"
EPOCHS = 20
BATCH_SIZE = 32
BENCHMARK_DIR = "../hdl-benchmarks/iscas85/bench/"

# =============================================================================
# PART 1: OPTIMIZED PARALLEL DATA GENERATION
# =============================================================================

def get_target_files():
    """Returns a list of .bench files in the benchmark directory."""
    if not os.path.exists(BENCHMARK_DIR):
        return []
    return [f for f in os.listdir(BENCHMARK_DIR) if f.endswith(".bench")]

def process_single_circuit(filename):
    """Worker function to process one circuit file."""
    filepath = os.path.join(BENCH_DIR, filename)
    local_dataset = []
    
    # Verbose logging to show progress
    print(f"[{filename}] Starting...", flush=True)
    
    try:
        miter = WireFaultMiter(filepath)
        if not miter.gates: return []
        
        # Use CPU for generation to avoid overhead
        extractor = VectorizedGraphExtractor(filepath, var_map=miter.var_map, device='cpu')
        input_set = set(miter.inputs)
        
        for i in range(SAMPLES_PER_FILE):
            target_gate = random.choice(miter.gates)[0]
            clauses = miter.build_miter(target_gate, None, 1)
            cnf = CNF()
            cnf.extend(clauses)
            
            # Solve once for Ground Truth
            with Glucose3(bootstrap_with=cnf) as solver:
                if not solver.solve(): continue
                model = solver.get_model()
                if not model: continue
                
                # Reuse solver for probing inputs (Fast)
                with Glucose3(bootstrap_with=cnf) as probe_solver:
                    current_conflicts = probe_solver.accum_stats()['conflicts']
                    input_importance = {}
                    input_polarity = {} 
                    
                    for input_name in miter.inputs:
                        var_id = miter.var_map[input_name]
                        correct_val = var_id if var_id in model else -var_id
                        wrong_val = -correct_val
                        
                        # Probe with Assumption (Reusing solver instance)
                        result = probe_solver.solve(assumptions=[wrong_val])
                        
                        new_conflicts = probe_solver.accum_stats()['conflicts']
                        delta = new_conflicts - current_conflicts
                        current_conflicts = new_conflicts
                        
                        if result:
                            importance = delta # Difficulty is effort
                        else:
                            importance = 10000 # Critical variable
                        
                        input_importance[input_name] = importance
                        
                        if importance > 0:
                            input_polarity[input_name] = 1.0 if var_id in model else 0.0
                        else:
                            input_polarity[input_name] = 0.5
                
                # Normalize
                max_imp = max(input_importance.values()) if input_importance else 1
                
                # Build Data
                data = extractor.get_data_for_fault(target_gate)
                y_polarity = torch.zeros(len(data.node_names), 1)
                y_importance = torch.zeros(len(data.node_names), 1)
                train_mask = torch.zeros(len(data.node_names), 1)
                
                for i, node_name in enumerate(data.node_names):
                    if node_name in input_set:
                        y_polarity[i] = input_polarity.get(node_name, 0.5)
                        y_importance[i] = input_importance.get(node_name, 0) / max(max_imp, 1)
                        train_mask[i] = 1.0
                
                data.y_polarity = y_polarity
                data.y_importance = y_importance
                data.train_mask = train_mask
                
                local_dataset.append(data)
    
    except Exception as e:
        print(f"[{filename}] Error: {e}", flush=True)
        return []

    print(f"[{filename}] Finished. Generated {len(local_dataset)} samples.", flush=True)
    return local_dataset

def generate_dataset():
    print(f"--- MINING DUAL-TASK ORACLE DATA (PARALLEL) ---")
    
    if not os.path.exists(BENCH_DIR):
        print(f"Error: {BENCH_DIR} not found.")
        return

    files = get_target_files()
    print(f"Found {len(files)} circuits. Starting workers...")

    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Limit workers to prevent freezing
    num_workers = min(4, os.cpu_count())
    dataset = []

    with mp.Pool(processes=num_workers) as pool:
        # imap_unordered makes the progress bar smoother
        results = list(tqdm(pool.imap_unordered(process_single_circuit, files), total=len(files)))
        
        for res in results:
            dataset.extend(res)

    print(f"--- Mining Complete. Collected {len(dataset)} samples. ---")
    torch.save(dataset, DATASET_PATH)


# =============================================================================
# PART 2: DUAL-TASK MODEL (FIXED)
# =============================================================================

class CircuitGNN_DualTask(torch.nn.Module):
    def __init__(self, num_node_features=16, num_layers=8, hidden_dim=64, dropout=0.2):
        super(CircuitGNN_DualTask, self).__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers # <--- FIXED MISSING ATTRIBUTE
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GATv2Conv(num_node_features, hidden_dim, heads=2, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        self.convs.append(GATv2Conv(hidden_dim, 32, heads=2, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(32))
        
        # Dual Heads
        self.importance_head = torch.nn.Linear(32, 1)
        self.polarity_head = torch.nn.Linear(32, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.convs[0](x, edge_index)
        x = self.bns[0](x)
        x = torch.nn.functional.elu(x)
        
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = torch.nn.functional.elu(x)
            x = x + identity
        
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = torch.nn.functional.elu(x)
        
        importance = self.importance_head(x)
        polarity = torch.sigmoid(self.polarity_head(x))
        return importance, polarity


# =============================================================================
# PART 3: TRAINING LOOP
# =============================================================================

def train_model():
    print("--- Training Dual-Task GNN ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found. Generating...")
        generate_dataset()
    
    dataset = torch.load(DATASET_PATH, weights_only=False)
    split = int(len(dataset) * 0.8)
    train_loader = DataLoader(dataset[:split], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset[split:], batch_size=BATCH_SIZE, shuffle=False)
    
    model = CircuitGNN_DualTask(num_node_features=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    crit_imp = nn.MSELoss(reduction='none')
    crit_pol = nn.BCELoss(reduction='none')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            p_imp, p_pol = model(batch)
            
            mask = batch.train_mask
            mask_sum = mask.sum().clamp(min=1)
            
            l_imp = (crit_imp(p_imp, batch.y_importance) * mask).sum() / mask_sum
            l_pol = (crit_pol(p_pol, batch.y_polarity) * mask).sum() / mask_sum
            
            loss = l_imp + l_pol
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")


# =============================================================================
# PART 4: BENCHMARKING (SET PHASES)
# =============================================================================

def solve_with_phases(cnf, hint_literals):
    """
    Solve using set_phases for soft guidance.
    hint_literals: List of signed integers.
    """
    with Glucose3(bootstrap_with=cnf) as solver:
        # Apply GNN hints as preferred polarities
        solver.set_phases(hint_literals)
        
        # Solve normally
        result = solver.solve()
        
        conflicts = solver.accum_stats()['conflicts']
        
    return result, conflicts


def run_benchmark():
    print(f"--- BENCHMARKING WITH SET_PHASES ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CircuitGNN_DualTask(num_node_features=16).to(device)
    if not os.path.exists(MODEL_PATH):
        print("Train model first.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    results = []
    files = get_target_files()
    
    for filename in files:
        filepath = os.path.join(BENCHMARK_DIR, filename)
        print(f"\nProcessing {filename}...")
        
        try:
            miter = WireFaultMiter(filepath)
            if not miter.gates: continue
            
            extractor = VectorizedGraphExtractor(filepath, var_map=miter.var_map, device=device.type)
            input_names = set(miter.inputs)
            
            for i in range(20): 
                target_gate = random.choice(miter.gates)[0]
                
                clauses = miter.build_miter(target_gate, None, 1)
                cnf = CNF()
                cnf.extend(clauses)
                
                # 1. Standard Solve (Baseline)
                t_start = time.time()
                with Glucose3(bootstrap_with=cnf) as s:
                    s.solve()
                    std_conflicts = s.accum_stats()['conflicts']
                std_time = time.time() - t_start
                
                # 2. GNN Inference
                t_gnn_start = time.time()
                data = extractor.get_data_for_fault(target_gate)
                data = data.to(device)
                
                with torch.no_grad():
                    imp_scores, pol_scores = model(data)
                
                # 3. Construct Hints
                candidates = []
                for idx, name in enumerate(data.node_names):
                    if name in input_names:
                        imp = imp_scores[idx].item()
                        prob = pol_scores[idx].item()
                        var_id = miter.var_map.get(name)
                        
                        if var_id:
                            # 1.0 -> True (var_id), 0.0 -> False (-var_id)
                            signed_lit = var_id if prob > 0.5 else -var_id
                            candidates.append((signed_lit, imp))
                
                # Sort by importance (highest first) 
                candidates.sort(key=lambda x: x[1], reverse=True)
                hint_literals = [x[0] for x in candidates]
                
                # 4. Guided Solve
                _, gnn_conflicts = solve_with_phases(cnf, hint_literals)
                gnn_time = time.time() - t_gnn_start
                
                speedup = std_conflicts / max(gnn_conflicts, 1)
                print(f"  Fault {target_gate}: {std_conflicts} -> {gnn_conflicts} ({speedup:.2f}x)")
                
                results.append({
                    "Circuit": filename,
                    "Speedup": speedup,
                    "Std_Conf": std_conflicts,
                    "GNN_Conf": gnn_conflicts,
                    "Time_Speedup": std_time / max(gnn_time, 0.0001)
                })
                
        except Exception as e:
            print(f"Error: {e}")

    if results:
        with open("results_set_phases.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("Saved to results_set_phases.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python importance_pipeline_16feat.py [generate|train|benchmark]")
    else:
        cmd = sys.argv[1]
        if cmd == "generate": generate_dataset()
        elif cmd == "train": train_model()
        elif cmd == "benchmark": run_benchmark()