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

# =============================================================================
# EXTRACTOR (Defined here to ensure completeness)
# =============================================================================
class VectorizedGraphExtractor:
    """
    High-Performance SCOAP Extractor using Vectorized Tensor Operations.
    Generates 16-dimensional feature vectors including Observability.
    """
    
    # Gate Type Mapping
    TYPE_MAP = {
        'INPUT': 0, 'PPI': 0, 
        'BUFF': 1, 'NOT': 2,
        'AND': 3, 'NAND': 4,
        'OR': 5, 'NOR': 6,
        'XOR': 7, 'XNOR': 7
    }

    def __init__(self, bench_path, var_map=None, device='cpu'):
        self.parser = BenchParser(bench_path)
        self.device = device
        
        # 1. Build Name Mappings (Sync with Miter if var_map provided)
        if var_map:
            self.var_map = var_map
        else:
            self.var_map = self.parser.build_var_map()
            
        self.ordered_names = sorted(self.var_map.keys(), key=lambda k: self.var_map[k])
        self.name_to_idx = {name: i for i, name in enumerate(self.ordered_names)}
        self.num_nodes = len(self.ordered_names)
        
        # 2. Build Structural Tensors
        self.edges_list = []
        self.node_types = torch.zeros(self.num_nodes, dtype=torch.long, device=device)
        
        # Assign Gate Types
        for name, g_type, _ in self.parser.gates:
            if name in self.name_to_idx:
                idx = self.name_to_idx[name]
                self.node_types[idx] = self.TYPE_MAP.get(g_type, 1) # Default to BUFF
        
        # Overwrite Types for Inputs/PPIs
        for pi in self.parser.inputs:
            if pi in self.name_to_idx:
                self.node_types[self.name_to_idx[pi]] = self.TYPE_MAP['INPUT']
        for ppi in self.parser.ppis:
            if ppi in self.name_to_idx:
                self.node_types[self.name_to_idx[ppi]] = self.TYPE_MAP['INPUT']
        
        # Build Edge List (Source -> Dest)
        for out, _, inputs in self.parser.gates:
            if out in self.name_to_idx:
                dst = self.name_to_idx[out]
                for inp in inputs:
                    if inp in self.name_to_idx:
                        src = self.name_to_idx[inp]
                        self.edges_list.append([src, dst])
        
        # Create Edge Index Tensor
        if self.edges_list:
            self.edge_index = torch.tensor(self.edges_list, dtype=torch.long, device=device).t().contiguous()
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            
        # Create Boolean Masks for Vectorized Logic
        self.masks = {}
        for t_name, t_id in self.TYPE_MAP.items():
            self.masks[t_name] = (self.node_types == t_id)

        # Pre-build Python adjacency for BFS traversals (Distance calculation)
        self.adj = [[] for _ in range(self.num_nodes)]       # Forward: src -> [dst]
        self.parents = [[] for _ in range(self.num_nodes)]  # Backward: dst -> [src]
        
        for src, dst in self.edges_list:
            self.adj[src].append(dst)
            self.parents[dst].append(src)
            
        # 3. Compute Metrics Immediately
        self.cc0, self.cc1, self.co = self._compute_scoap_vectorized()
        self.x_base = self._build_base_features()

    def _compute_scoap_vectorized(self):
        """Vectorized SCOAP: Forward Controllability & Backward Observability"""
        num_nodes = self.num_nodes
        src_idx, dst_idx = self.edge_index
        
        # --- Part A: Controllability (Forward) ---
        cc0 = torch.ones(num_nodes, device=self.device)
        cc1 = torch.ones(num_nodes, device=self.device)
        
        mask_and = self.masks['AND'] | self.masks['NAND']
        mask_or  = self.masks['OR'] | self.masks['NOR']
        mask_inv = self.masks['NAND'] | self.masks['NOR'] | self.masks['NOT']
        mask_xor = self.masks['XOR']
        mask_buf_not = self.masks['BUFF'] | self.masks['NOT']
        
        for _ in range(50): 
            cc0_prev, cc1_prev = cc0.clone(), cc1.clone()
            
            edge_cc0 = cc0[src_idx]
            edge_cc1 = cc1[src_idx]
            
            # Aggregate per Gate (Destination)
            min_cc0 = torch.zeros(num_nodes, device=self.device).scatter_reduce_(
                0, dst_idx, edge_cc0, reduce='min', include_self=False)
            min_cc1 = torch.zeros(num_nodes, device=self.device).scatter_reduce_(
                0, dst_idx, edge_cc1, reduce='min', include_self=False)
            
            sum_cc0 = torch.zeros(num_nodes, device=self.device).scatter_add_(0, dst_idx, edge_cc0)
            sum_cc1 = torch.zeros(num_nodes, device=self.device).scatter_add_(0, dst_idx, edge_cc1)
            
            # Apply Logic
            cc0[mask_and] = min_cc0[mask_and] + 1
            cc1[mask_and] = sum_cc1[mask_and] + 1
            
            cc0[mask_or] = sum_cc0[mask_or] + 1
            cc1[mask_or] = min_cc1[mask_or] + 1
            
            cc0[mask_buf_not] = min_cc0[mask_buf_not] + 1
            cc1[mask_buf_not] = min_cc1[mask_buf_not] + 1
            
            cc0[mask_xor] = torch.minimum(sum_cc0[mask_xor], sum_cc1[mask_xor]) + 1
            cc1[mask_xor] = torch.maximum(min_cc0[mask_xor], min_cc1[mask_xor]) + 1

            # Inversions
            temp_cc0 = cc0.clone()
            cc0[mask_inv] = cc1[mask_inv]
            cc1[mask_inv] = temp_cc0[mask_inv]
            
            # Reset Inputs
            mask_input = self.masks['INPUT']
            cc0[mask_input] = 1.0
            cc1[mask_input] = 1.0
            
            if torch.allclose(cc0, cc0_prev) and torch.allclose(cc1, cc1_prev):
                break

        # --- Part B: Observability (Backward) ---
        co = torch.full((num_nodes,), 1e6, device=self.device)
        
        output_indices = [self.name_to_idx[n] for n in self.parser.all_outputs if n in self.name_to_idx]
        if output_indices:
            co[torch.tensor(output_indices, device=self.device)] = 0.0

        gate_cc0_sum = torch.zeros(num_nodes, device=self.device).scatter_add_(0, dst_idx, cc0[src_idx])
        gate_cc1_sum = torch.zeros(num_nodes, device=self.device).scatter_add_(0, dst_idx, cc1[src_idx])
        gate_min_sum = torch.zeros(num_nodes, device=self.device).scatter_add_(
            0, dst_idx, torch.minimum(cc0[src_idx], cc1[src_idx]))

        for _ in range(50):
            co_prev = co.clone()
            
            co_dst = co[dst_idx]
            dst_types = self.node_types[dst_idx]
            side_costs = torch.zeros_like(co_dst)
            
            # Side input logic
            is_and = (dst_types == self.TYPE_MAP['AND']) | (dst_types == self.TYPE_MAP['NAND'])
            side_costs[is_and] = gate_cc1_sum[dst_idx][is_and] - cc1[src_idx][is_and]
            
            is_or = (dst_types == self.TYPE_MAP['OR']) | (dst_types == self.TYPE_MAP['NOR'])
            side_costs[is_or] = gate_cc0_sum[dst_idx][is_or] - cc0[src_idx][is_or]
            
            is_xor = (dst_types == self.TYPE_MAP['XOR'])
            side_costs[is_xor] = gate_min_sum[dst_idx][is_xor] - torch.minimum(cc0[src_idx], cc1[src_idx])[is_xor]
            
            path_costs = co_dst + side_costs + 1
            
            new_co = torch.zeros_like(co).scatter_reduce_(
                0, src_idx, path_costs, reduce='min', include_self=False
            )
            
            co = torch.minimum(co, new_co)
            
            if torch.allclose(co, co_prev):
                break
                
        return cc0, cc1, co

    def _compute_depth_fast(self, reverse=False):
        """Vectorized Topological Depth"""
        d_vals = torch.zeros(self.num_nodes, device=self.device)
        src_idx, dst_idx = self.edge_index
        prop_src = dst_idx if reverse else src_idx
        prop_dst = src_idx if reverse else dst_idx
        
        for _ in range(50):
            changed = False
            src_depths = d_vals[prop_src]
            new_depths = torch.zeros(self.num_nodes, device=self.device).scatter_reduce_(
                0, prop_dst, src_depths, reduce='amax', include_self=True
            )
            new_depths = new_depths + 1
            if not torch.allclose(d_vals, new_depths):
                d_vals = new_depths
                changed = True
            if not changed: break
            
        max_d = d_vals.max() if d_vals.max() > 0 else 1.0
        return (d_vals / max_d).unsqueeze(1)

    def _build_base_features(self):
        """
        Builds 16-dimensional feature matrix
        [0-7]: Type, [8-9]: Depth, [10-11]: Fault, [12-14]: SCOAP, [15]: Output
        """
        x_type = torch.nn.functional.one_hot(self.node_types, num_classes=8).float()
        fwd_depth = self._compute_depth_fast(reverse=False)
        rev_depth = self._compute_depth_fast(reverse=True)
        
        f_cc0 = torch.log(self.cc0 + 1).unsqueeze(1) / 10.0
        f_cc1 = torch.log(self.cc1 + 1).unsqueeze(1) / 10.0
        f_co  = torch.log(self.co + 1).unsqueeze(1) / 10.0
        
        is_output = torch.zeros((self.num_nodes, 1), device=self.device)
        for name in self.parser.all_outputs:
            if name in self.name_to_idx:
                is_output[self.name_to_idx[name]] = 1.0
                
        zeros = torch.zeros((self.num_nodes, 2), device=self.device)
        
        return torch.cat([x_type, fwd_depth, rev_depth, zeros, f_cc0, f_cc1, f_co, is_output], dim=1)

    def get_data_for_fault(self, fault_name):
        """Generate Data object for a specific fault"""
        x = self.x_base.clone()
        tid = self.name_to_idx.get(fault_name)
        
        if tid is not None:
            x[tid, 10] = 1.0 # Mark target
            
            # BFS for Distance (Index 11)
            dist = torch.full((self.num_nodes,), -1.0, device=self.device)
            dist[tid] = 0.0
            queue = [tid]
            visited = {tid: 0}
            idx = 0
            
            while idx < len(queue):
                u = queue[idx]
                idx += 1
                d = visited[u]
                if d >= 10: continue
                
                neighbors = self.adj[u] + self.parents[u]
                for v in neighbors:
                    if v not in visited:
                        visited[v] = d + 1
                        dist[v] = d + 1
                        queue.append(v)
            
            mask_visited = (dist != -1)
            if mask_visited.any():
                max_d = dist.max()
                if max_d == 0: max_d = 1.0
                x[mask_visited, 11] = 1.0 - (dist[mask_visited] / max_d)
                
        return Data(x=x, edge_index=self.edge_index, node_names=self.ordered_names)

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
# PART 1: OPTIMIZED PARALLEL DATA GENERATION (CONE BASED)
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
    # Sort largest first for better scheduling
    return sorted(file_list, key=lambda x: os.path.getsize(os.path.join(DIR, x)), reverse=True)

def process_single_circuit(filename):
    """Worker using CONE SPLITTING to handle Large Circuits."""
    set_global_seed(SEED + len(filename)) 
    filepath = os.path.join(GENERATE_TRAIN_DATA_DIR, filename)
    local_dataset = []

    try:
        miter = WireFaultMiter(filepath)
        if not miter.gates: return []
        
        # Adjust samples based on size (Giant circuits get fewer samples, but higher quality)
        num_gates = len(miter.gates)
        local_samples = 10 if num_gates > 5000 else SAMPLES_PER_FILE
        
        extractor = VectorizedGraphExtractor(filepath, var_map=miter.parser.build_var_map(), device='cpu')
        
        print(f"[{filename}] Mining {local_samples} samples (Cone Optimized)...", flush=True)

        for i in range(local_samples):
            # 1. Pick a Fault
            target_gate = random.choice(miter.gates)[0]
            
            # 2. Extract a Valid Cone (Instead of solving whole circuit)
            outs = miter.get_reachable_outputs(target_gate)
            if not outs: continue
            
            target_out = random.choice(outs)
            cone_gates = miter.get_logic_cone([target_out], target_gate)
            
            # 3. Swap Gates & Build Miter
            orig_gates = miter.gates
            miter.gates = cone_gates
            
            clauses = miter.build_miter(target_gate, None, 1) # SA1
            miter.gates = orig_gates # Restore immediately
            
            with Glucose3(bootstrap_with=clauses) as solver:
                solver.conf_budget(5000)
                if not solver.solve(): continue
                
                model = solver.get_model()
                if not model: continue
                
                # 4. Probe Inputs (Only those in the Cone!)
                cone_inputs = set()
                for _, _, inps in cone_gates:
                    for inp in inps:
                        if inp in miter.inputs: cone_inputs.add(inp)
                
                probe_list = list(cone_inputs)
                if len(probe_list) > 50: probe_list = random.sample(probe_list, 50)
                
                with Glucose3(bootstrap_with=clauses) as probe:
                    base_conf = probe.accum_stats()['conflicts']
                    input_importance = {}
                    input_polarity = {}
                    
                    for input_name in probe_list:
                        if input_name not in miter.var_map: continue
                        var_id = miter.var_map[input_name]
                        
                        correct = var_id if var_id in model else -var_id
                        probe.conf_budget(500)
                        res = probe.solve(assumptions=[-correct])
                        new_conf = probe.accum_stats()['conflicts']
                        
                        imp = new_conf - base_conf
                        base_conf = new_conf
                        
                        input_importance[input_name] = imp if res else 2000
                        input_polarity[input_name] = 1.0 if var_id in model else 0.0

                    # 5. Build Data Object
                    if input_importance:
                        data = extractor.get_data_for_fault(target_gate)
                        max_imp = max(input_importance.values())
                        
                        y_pol = torch.zeros(len(data.node_names), 1)
                        y_imp = torch.zeros(len(data.node_names), 1)
                        mask = torch.zeros(len(data.node_names), 1)
                        
                        for k, node in enumerate(data.node_names):
                            if node in input_importance:
                                y_pol[k] = input_polarity[node]
                                y_imp[k] = input_importance[node] / max(max_imp, 1)
                                mask[k] = 1.0
                        
                        data.y_polarity = y_pol
                        data.y_importance = y_imp
                        data.train_mask = mask
                        local_dataset.append(data)

    except Exception as e:
        print(f"[{filename}] Error: {e}", flush=True)
        return []

    return local_dataset

def generate_dataset():
    print(f"--- MINING DATA (CONE OPTIMIZED) ---")
    if not os.path.exists(GENERATE_TRAIN_DATA_DIR): return

    files = get_target_files(GENERATE_TRAIN_DATA_DIR)
    dataset = []
    
    try: mp.set_start_method('spawn', force=True)
    except: pass

    with mp.Pool(min(4, os.cpu_count())) as pool:
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
    if not os.path.exists(DATASET_PATH): 
        print("Dataset not found. Generating...")
        generate_dataset()
    
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
# PART 4: DETERMINISTIC BENCHMARKING (OPTIMIZED)
# =============================================================================

def run_benchmark():
    print(f"--- BENCHMARKING (GNN + CONE SPLIT + BRANCHLESS) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CircuitGNN_DualTask(num_node_features=16).to(device)
    if not os.path.exists(MODEL_PATH):
        print("Train model first.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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
            
            # Run 10 faults
            for i in range(10): 
                target_gate = random.choice(all_gates)[0]
                
                # 1. Inference
                t_gnn_start = time.time()
                data = extractor.get_data_for_fault(target_gate)
                data = data.to(device)
                
                with torch.no_grad():
                    imp_scores, pol_scores = model(data)
                
                # 2. Extract Hints
                hints = []
                for idx, name in enumerate(data.node_names):
                    if name in miter.inputs:
                        imp = imp_scores[idx].item()
                        prob = pol_scores[idx].item()
                        var_id = miter.var_map.get(name)
                        if var_id:
                            val = 1 if prob > 0.5 else 0
                            # Map 0 -> -Lit, 1 -> Lit
                            hints.append((var_id if val else -var_id, imp))
                            
                hints.sort(key=lambda x: -x[1])
                hint_lits = [h[0] for h in hints]
                
                # 3. Solve using Optimized Cone Strategy with Hints
                assignment = miter.solve_fault_specific_cones(target_gate, gnn_hints=hint_lits)
                
                dur = time.time() - t_gnn_start
                status = "DETECTED" if assignment else "UNSAT/UNDETECTED"
                print(f"  Fault {target_gate}: {status} in {dur:.4f}s")
                if assignment:
                    # Print Compact Vector
                    vec = "".join([str(assignment.get(k, 'X')) for k in sorted(miter.inputs)])
                    print(f"  Vector: {vec[:50]}...")
                
                results.append({
                    "Circuit": filename,
                    "Fault": target_gate,
                    "Status": status,
                    "Time": dur
                })
                
        except Exception as e:
            print(f"Error: {e}")

    if results:
        with open("results_optimized.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("Saved to results_optimized.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_train_bench_mem_efficient.py [generate|train|benchmark]")
    else:
        cmd = sys.argv[1]
        if cmd == "generate": generate_dataset()
        elif cmd == "train": train_model()
        elif cmd == "benchmark": run_benchmark()