"""
Complete pipeline for Dual-Task GNN training with Parallel Generation & Polarity Guidance.
Uses Cone Splitting and Branch-less Stem Optimization for Large Circuits.
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
DATASET_PATH = "dataset_oracle_dual_17feat.pt"
SAMPLES_PER_FILE = 50
MODEL_PATH = "gnn_model_dual_task_17feat.pth"
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

def get_target_files(DIR):
    if not os.path.exists(DIR): return []
    file_list = []
    for root, dirs, files in os.walk(DIR):
        for f in files:
            if f.endswith(".bench"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, DIR)
                file_list.append(rel_path)
    return sorted(file_list, key=lambda x: os.path.getsize(os.path.join(DIR, x)), reverse=True)

def process_single_circuit(filename):
    """Worker passing fault_type=1 to extractor"""
    set_global_seed(SEED + len(filename)) 
    filepath = os.path.join(GENERATE_TRAIN_DATA_DIR, filename)
    local_dataset = []
    try:
        miter = WireFaultMiter(filepath)
        if not miter.gates: return []
        local_samples = 10 if len(miter.gates) > 5000 else SAMPLES_PER_FILE
        print(f"[{filename}] Mining {local_samples} samples (Cone Optimized)...", flush=True)
        extractor = VectorizedGraphExtractor(filepath, var_map=miter.parser.build_var_map(), device='cpu')
        
        for i in range(local_samples):
            # 1. Pick a Fault
            target_gate = random.choice(miter.gates)[0]
            outs = miter.get_reachable_outputs(target_gate)
            if not outs: continue
            
            # --- FIX FOR HANG ON B17/B19 ---
            # Don't check all outputs. Sample 20 random ones.
            sample_outs = random.sample(outs, min(20, len(outs)))
            candidates = [(len(miter.get_logic_cone([o], target_gate)), o) for o in sample_outs]
            candidates.sort()
            target_out = candidates[0][1]
            
            cone_gates = miter.get_logic_cone([target_out], target_gate)
            
            orig_gates = miter.gates; miter.gates = cone_gates
            
            # --- FAULT TYPE IS 1 (SA1) ---
            FAULT_TYPE = 1 
            clauses = miter.build_miter(target_gate, None, FAULT_TYPE) 
            miter.gates = orig_gates
            
            with Glucose3(bootstrap_with=clauses) as solver:
                solver.conf_budget(5000)
                if solver.solve():
                    model = solver.get_model()
                    cone_inputs = set([i for _,_,inps in cone_gates for i in inps if i in miter.inputs])
                    probe_list = list(cone_inputs)[:50]
                    
                    with Glucose3(bootstrap_with=clauses) as probe:
                        base = probe.accum_stats()['conflicts']
                        input_importance = {}
                        input_polarity = {} 
                        for inp in probe_list:
                            if inp not in miter.var_map: continue
                            vid = miter.var_map[inp]
                            val = vid if vid in model else -vid
                            probe.conf_budget(500)
                            res = probe.solve(assumptions=[-val])
                            new_c = probe.accum_stats()['conflicts']
                            input_importance[inp] = (new_c - base) if res else 2000
                            input_polarity[inp] = 1.0 if vid in model else 0.0
                            base = new_c
                    
                    if input_importance:
                        # --- PASS FAULT TYPE HERE ---
                        data = extractor.get_data_for_fault(target_gate, fault_type=FAULT_TYPE)
                        max_imp = max(input_importance.values())
                        y_pol = torch.zeros(len(data.node_names), 1)
                        y_imp = torch.zeros(len(data.node_names), 1)
                        mask = torch.zeros(len(data.node_names), 1)
                        for k, n in enumerate(data.node_names):
                            if n in input_importance:
                                y_pol[k] = input_polarity[n]
                                y_imp[k] = input_importance[n] / max(max_imp, 1)
                                mask[k] = 1.0
                        data.y_polarity = y_pol
                        data.y_importance = y_imp
                        data.train_mask = mask
                        local_dataset.append(data)
                        
                        # Progress print
                        if i % 2 == 0: 
                            print(f"[{filename}] {i+1}/{local_samples} done.", flush=True)

    except Exception as e:
        print(f"[{filename}] Error: {e}")
    return local_dataset

def generate_dataset():
    files = get_target_files(GENERATE_TRAIN_DATA_DIR)
    dataset = []
    try: mp.set_start_method('spawn', force=True)
    except: pass
    with mp.Pool(min(4, os.cpu_count())) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_circuit, files), total=len(files)))
        for res in results: dataset.extend(res)
    torch.save(dataset, DATASET_PATH)

# =============================================================================
# PART 2 & 3: MODEL AND TRAINING
# =============================================================================

class CircuitGNN_DualTask(torch.nn.Module):
    # --- UPDATED INPUT DIM TO 17 ---
    def __init__(self, num_node_features=17, num_layers=20, hidden_dim=64, dropout=0.2):
        super(CircuitGNN_DualTask, self).__init__()
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
        x = torch.relu(self.bns[0](self.convs[0](x, edge_index)))
        for i in range(1, len(self.convs) - 1):
            x = x + torch.relu(self.bns[i](self.convs[i](x, edge_index)))
        x = torch.relu(self.bns[-1](self.convs[-1](x, edge_index)))
        return self.importance_head(x), torch.sigmoid(self.polarity_head(x))

def train_model():
    print("--- Training Dual-Task GNN ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(DATASET_PATH):
        generate_dataset()

    dataset = torch.load(DATASET_PATH, weights_only=False)
    loader = DataLoader(dataset[:int(len(dataset)*0.8)],
                        batch_size=BATCH_SIZE,
                        shuffle=True)

    model = CircuitGNN_DualTask(num_node_features=17).to(device)

    # LOAD MODEL IF IT EXISTS
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("No existing model found. Training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    crit_imp = nn.MSELoss(reduction='none')
    crit_pol = nn.BCELoss(reduction='none')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in loader:
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

        print(f"Epoch {epoch+1}: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


# =============================================================================
# PART 4: DETERMINISTIC BENCHMARKING (FIXED WITH OPTIMIZATION)
# =============================================================================

def run_benchmark():
    print(f"--- BENCHMARKING (Cone Split + Branchless + GNN) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # --- UPDATE MODEL CALL TO 17 ---
    model = CircuitGNN_DualTask(num_node_features=17).to(device)
    if os.path.exists(MODEL_PATH): model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    results = []
    files = get_target_files(BENCHMARK_DIR)
    random.seed(SEED) 
    
    for filename in files:
        filepath = os.path.join(BENCHMARK_DIR, filename)
        print(f"\nProcessing {filename}...")
        try:
            miter = WireFaultMiter(filepath)
            if not miter.gates: continue
            extractor = VectorizedGraphExtractor(filepath, var_map=miter.var_map, device=device.type)
            
            all_gates = sorted(miter.gates, key=lambda x: x[0])
            for i in range(10): 
                target_gate = random.choice(all_gates)[0]
                FAULT_TYPE = 1 # SA1
                
                # --- GNN INFERENCE ---
                t0 = time.time()
                # Pass fault type 1
                data = extractor.get_data_for_fault(target_gate, fault_type=FAULT_TYPE).to(device)
                with torch.no_grad(): imp, pol = model(data)
                
                hints = {}
                for idx, name in enumerate(data.node_names):
                    if name in miter.inputs: hints[name] = pol[idx].item()
                
                # --- 1. BASELINE RUN (No Hints) ---
                _, conf_std = miter.solve_fault_specific_cones(target_gate, FAULT_TYPE, gnn_hints=None)
                
                # --- 2. GNN RUN (With Hints) ---
                assign_gnn, conf_gnn = miter.solve_fault_specific_cones(target_gate, FAULT_TYPE, gnn_hints=hints)
                dur = time.time() - t0
                
                spd = conf_std / max(conf_gnn, 1)
                status = "DETECTED" if assign_gnn else "UNSAT"
                
                if assign_gnn:
                    vec_str = "".join([str(assign_gnn.get(k, 'X')) for k in sorted(miter.inputs)])
                    print(f"  {target_gate}: {status} | Conf: {conf_std} -> {conf_gnn} ({spd:.2f}x)")
                    print(f"  Vector: {vec_str[:30]}...")
                else:
                    vec_str = "UNSAT"
                    print(f"  {target_gate}: {status} | Conf: {conf_std} -> {conf_gnn} ({spd:.2f}x)")
                
                results.append({
                    "Circuit": filename, "Fault": target_gate, "Speedup": spd,
                    "Std_Conf": conf_std, "GNN_Conf": conf_gnn, "Status": status,
                    "Vector": vec_str
                })
        except Exception as e:
            print(f"Error: {e}")

    if results:
        with open("results_optimized.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python data_train_bench_mem_efficient.py [generate|train|benchmark]")
    else:
        cmd = sys.argv[1]
        if cmd == "generate": generate_dataset()
        elif cmd == "train": train_model()
        elif cmd == "benchmark": run_benchmark()