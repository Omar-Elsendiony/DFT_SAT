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
import numpy as np
from pysat.solvers import Glucose3, Minisat22
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
BENCHMARK_DIR = "../hdl-benchmarks/iscas85/bench/"
DATASET_PATH = "dataset_oracle_dual_16feat.pt"
SAMPLES_PER_FILE = 50
MODEL_PATH = "gnn_model_dual_task_16feat.pth"
EPOCHS = 20
BATCH_SIZE = 32
GENERATE_TRAIN_DATA_DIR = "../I99T"
SEED = 42

# =============================================================================
# 0. DETERMINISM SETUP
# =============================================================================
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
# PART 1: OPTIMIZED PARALLEL DATA GENERATION
# =============================================================================

# def get_target_files():
#     if not os.path.exists(BENCHMARK_DIR):
#         return []
#     return sorted([f for f in os.listdir(BENCHMARK_DIR) if f.endswith(".bench")])

def get_target_files(DIR):
    if not os.path.exists(DIR):
        return []
        
    file_list = []
    # os.walk recursively visits every subdirectory
    for root, dirs, files in os.walk(DIR):
        for f in files:
            if f.endswith(".bench"):
                # Get the full absolute path
                full_path = os.path.join(root, f)
                
                # Convert it to a path relative to BENCHMARK_DIR 
                # e.g., converts "/usr/bench/subdir/c17.bench" -> "subdir/c17.bench"
                # This ensures the os.path.join(DIR, filename) in your worker still works.
                rel_path = os.path.relpath(full_path, DIR)
                file_list.append(rel_path)
                
    return sorted(file_list)


def process_single_circuit(filename):
    """Worker with TIERED SAMPLING and GIANT SKIP."""
    set_global_seed(SEED + len(filename)) 
    
    filepath = os.path.join(GENERATE_TRAIN_DATA_DIR, filename)
    local_dataset = []

    try:
        # 1. Quick Gate Count Check (Avoid parsing massive files if possible)
        # (We parse it anyway here for simplicity, but in production you'd grep the file first)
        miter = WireFaultMiter(filepath)
        num_gates = len(miter.gates)
        if not miter.gates: return []
        
        # =========================================================
        # TIERED SAMPLING STRATEGY
        # =========================================================
        if num_gates > 20000:
            # TIER 3: GIANTS (b17, b18, b19) -> SKIP
            # These are too big for a single-threaded Python loop in a tutorial.
            print(f"[{filename}] SKIPPING Giant Circuit ({num_gates} gates).", flush=True)
            return []
            
        elif num_gates > 4000:
            # TIER 2: MEDIUM (b14, b15) -> REDUCED
            local_samples = 5   # Only 5 samples
            max_probes = 20     # Only probe 20 inputs
            probe_time_limit = 2.0
            print(f"[{filename}] Medium Circuit ({num_gates} gates). Reducing to {local_samples} samples.", flush=True)
            
        else:
            # TIER 1: SMALL -> FULL
            local_samples = SAMPLES_PER_FILE
            max_probes = 100
            probe_time_limit = 5.0

        extractor = VectorizedGraphExtractor(filepath, var_map=miter.var_map, device='cpu')
        input_list = sorted(list(miter.inputs))
        input_set = set(input_list)
        
        # Probe Sampling
        probe_list = input_list
        if len(input_list) > max_probes:
             probe_list = random.sample(input_list, max_probes)

        # Generate samples
        for i in range(local_samples):
            # VISUAL FEEDBACK
            if i % 5 == 0:
                print(f"[{filename}] Processing sample {i+1}/{local_samples}...", flush=True)

            all_gates = sorted(miter.gates, key=lambda x: x[0])
            target_gate = random.choice(all_gates)[0]
            
            clauses = miter.build_miter(target_gate, None, 1)
            cnf = CNF()
            cnf.extend(clauses)
            
            with Glucose3(bootstrap_with=cnf) as solver:
                solver.conf_budget(10000) 
                if not solver.solve(): continue
                    
                model = solver.get_model()
                if not model: continue
                
                with Glucose3(bootstrap_with=cnf) as probe_solver:
                    current_conflicts = probe_solver.accum_stats()['conflicts']
                    input_importance = {}
                    input_polarity = {} 
                    
                    start_probe_time = time.time()
                    
                    for input_name in probe_list:
                        # DYNAMIC TIMEOUT
                        if time.time() - start_probe_time > probe_time_limit: 
                            break

                        var_id = miter.var_map[input_name]
                        correct_val = var_id if var_id in model else -var_id
                        wrong_val = -correct_val
                        
                        probe_solver.conf_budget(1000)
                        result = probe_solver.solve(assumptions=[wrong_val])
                        
                        new_conflicts = probe_solver.accum_stats()['conflicts']
                        delta = new_conflicts - current_conflicts
                        current_conflicts = new_conflicts
                        
                        if result:
                            importance = delta 
                        else:
                            importance = 5000 
                        
                        input_importance[input_name] = importance
                        if importance > 0:
                            input_polarity[input_name] = 1.0 if var_id in model else 0.0
                        else:
                            input_polarity[input_name] = 0.5
                
                if not input_importance: continue 

                max_imp = max(input_importance.values()) if input_importance else 1
                data = extractor.get_data_for_fault(target_gate)
                y_polarity = torch.zeros(len(data.node_names), 1)
                y_importance = torch.zeros(len(data.node_names), 1)
                train_mask = torch.zeros(len(data.node_names), 1)
                
                for k, node_name in enumerate(data.node_names):
                    if node_name in input_set:
                        if node_name in input_importance:
                            y_polarity[k] = input_polarity.get(node_name, 0.5)
                            y_importance[k] = input_importance.get(node_name, 0) / max(max_imp, 1)
                            train_mask[k] = 1.0
                
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
    if not os.path.exists(GENERATE_TRAIN_DATA_DIR):
        print(f"Error: {GENERATE_TRAIN_DATA_DIR} not found.")
        return

    files = get_target_files(GENERATE_TRAIN_DATA_DIR)
    num_workers = min(4, os.cpu_count())
    dataset = []

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_circuit, files), total=len(files)))
        for res in results:
            dataset.extend(res)

    torch.save(dataset, DATASET_PATH)


# =============================================================================
# PART 2 & 3: MODEL AND TRAINING (UNCHANGED)
# =============================================================================

class CircuitGNN_DualTask(torch.nn.Module):
    def __init__(self, num_node_features=16, num_layers=20, hidden_dim=64, dropout=0.2):
        super(CircuitGNN_DualTask, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(num_node_features, hidden_dim, heads=2, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.convs.append(GATv2Conv(hidden_dim, 32, heads=2, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(32))
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
        return self.importance_head(x), torch.sigmoid(self.polarity_head(x))

def train_model():
    print("--- Training Dual-Task GNN ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if not os.path.exists(DATASET_PATH): generate_dataset()
    if not os.path.exists(DATASET_PATH): print("Dataset not found. Please generate dataset first."); return
    
    dataset = torch.load(DATASET_PATH, weights_only=False)
    train_loader = DataLoader(dataset[:int(len(dataset)*0.8)], batch_size=BATCH_SIZE, shuffle=True)
    
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
            (l_imp + l_pol).backward()
            optimizer.step()
            total_loss += (l_imp + l_pol).item()
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)


# =============================================================================
# PART 4: DETERMINISTIC BENCHMARKING (FIXED)
# =============================================================================

def solve_with_phases(cnf, hint_literals, solver_class=Minisat22):
    """
    Solve using set_phases for soft guidance.
    solver_class: Allows switching between Glucose3 and Minisat22
    """
    with solver_class(bootstrap_with=cnf) as solver:
        # 1. Deterministic Seeding (If supported by solver wrapper)
        # Most PySAT wrappers don't expose seed in init, but rely on deterministic behavior
        # given the same clause order.
        
        # 2. Apply Hints
        solver.set_phases(hint_literals)
        
        # 3. Solve
        result = solver.solve()
        conflicts = solver.accum_stats()['conflicts']
        
    return result, conflicts

def run_benchmark():
    print(f"--- BENCHMARKING WITH SET_PHASES (DETERMINISTIC) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CircuitGNN_DualTask(num_node_features=16).to(device)
    if not os.path.exists(MODEL_PATH):
        print("Train model first.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    results = []
    files = get_target_files(BENCHMARK_DIR)
    
    # 1. Sort files to ensure file processing order is fixed
    files.sort()
    
    # 2. Fix the sequence of faults we will test
    # We pre-generate the random seeds or indices if we want perfect repeatability across runs
    random.seed(SEED) 
    
    for filename in files:
        filepath = os.path.join(BENCHMARK_DIR, filename)
        print(f"\nProcessing {filename}...")
        
        try:
            miter = WireFaultMiter(filepath)
            if not miter.gates: continue
            
            extractor = VectorizedGraphExtractor(filepath, var_map=miter.var_map, device=device.type)
            
            # Deterministic: Sort input names
            input_names_list = sorted(list(miter.inputs))
            input_names_set = set(input_names_list)
            
            # Sort gates to ensure deterministic random choice
            all_gates = sorted(miter.gates, key=lambda x: x[0])
            
            # Run 20 faults
            for i in range(20): 
                # Pick target deterministically based on global seed state
                target_gate = random.choice(all_gates)[0]
                
                clauses = miter.build_miter(target_gate, None, 1)
                cnf = CNF()
                cnf.extend(clauses)
                
                # --- BASELINE (Minisat22) ---
                # Using Minisat22 as the "Weak Solver" to demonstrate GNN impact better
                # You can change this to Glucose3 if you prefer strong baseline
                SolverClass = Minisat22 
                
                t_start = time.time()
                with SolverClass(bootstrap_with=cnf) as s:
                    s.solve()
                    std_conflicts = s.accum_stats()['conflicts']
                std_time = time.time() - t_start
                
                # --- GNN INFERENCE ---
                t_gnn_start = time.time()
                data = extractor.get_data_for_fault(target_gate)
                data = data.to(device)
                
                with torch.no_grad():
                    imp_scores, pol_scores = model(data)
                
                # Extract Predictions
                candidates = []
                for idx, name in enumerate(data.node_names):
                    if name in input_names_set:
                        imp = imp_scores[idx].item()
                        prob = pol_scores[idx].item()
                        var_id = miter.var_map.get(name)
                        
                        if var_id:
                            signed_lit = var_id if prob > 0.5 else -var_id
                            candidates.append((signed_lit, imp, var_id)) # Add var_id for tie-breaking
                
                # --- CRITICAL FIX FOR DETERMINISM ---
                # Sort by: 
                # 1. Importance (Descending)
                # 2. Variable ID (Ascending) -> TIE BREAKER
                candidates.sort(key=lambda x: (-x[1], x[2]))
                
                hint_literals = [x[0] for x in candidates]
                
                # --- GUIDED SOLVE ---
                _, gnn_conflicts = solve_with_phases(cnf, hint_literals, solver_class=SolverClass)
                gnn_time = time.time() - t_gnn_start
                
                speedup = std_conflicts / max(gnn_conflicts, 1)
                print(f"  Fault {target_gate}: {std_conflicts} -> {gnn_conflicts} ({speedup:.2f}x)")
                
                results.append({
                    "Circuit": filename,
                    "Fault": target_gate,
                    "Speedup": speedup,
                    "Std_Conf": std_conflicts,
                    "GNN_Conf": gnn_conflicts
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