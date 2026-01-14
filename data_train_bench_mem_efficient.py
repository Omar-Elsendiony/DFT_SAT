"""
Complete pipeline for importance-aware GNN training with SCOAP Observability
This is the COMPLETE SECOND SCRIPT with 16 features (including CO and is_output)
"""

import os
import sys

# Add cloned PySAT to path BEFORE other imports
# This ensures we use the local clone, not pip-installed version
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

# =============================================================================
# CONFIGS
# =============================================================================
BENCH_DIR = "../hdl-benchmarks/iscas85/bench/"
DATASET_PATH = "dataset_oracle_importance_16feat.pt"
SAMPLES_PER_FILE = 50
MODEL_PATH = "gnn_model_importance_aware_16feat.pth"
EPOCHS = 20
BATCH_SIZE = 32
BENCHMARK_DIR = "../hdl-benchmarks/iscas85/bench/"

# =============================================================================
# PART 0: ENHANCED FAST GRAPH EXTRACTOR WITH OBSERVABILITY (16 FEATURES)
# =============================================================================

class FastGraphExtractor:
    """
    Graph extractor with FULL SCOAP (CC0, CC1, CO) and output marking
    Produces 16-dimensional node features
    """
    
    def __init__(self, bench_path, var_map=None):
        self.parser = BenchParser(bench_path)
        
        if var_map is None:
            var_map = self.parser.build_var_map()
        self.var_map = var_map
        
        self.ordered_names = sorted(var_map.keys(), key=lambda k: var_map[k])
        self.name_to_idx = {name: i for i, name in enumerate(self.ordered_names)}
        self.num_nodes = len(self.ordered_names)
        
        self.edges_list = []
        self.gate_types = {}
        self.gate_inputs = {i: [] for i in range(self.num_nodes)}
        self.gate_outputs = {i: [] for i in range(self.num_nodes)}
        
        for out, g_type, inputs in self.parser.gates:
            self.gate_types[out] = g_type
            if out in self.name_to_idx:
                dst = self.name_to_idx[out]
                for inp in inputs:
                    if inp in self.name_to_idx:
                        src = self.name_to_idx[inp]
                        self.edges_list.append([src, dst])
                        self.gate_inputs[dst].append(src)
                        self.gate_outputs[src].append(dst)
        
        for ppi in self.parser.ppis:
            if ppi in self.name_to_idx:
                self.gate_types[ppi] = 'PPI'
        
        for pi in self.parser.inputs:
            if pi not in self.gate_types and pi in self.name_to_idx:
                self.gate_types[pi] = 'INPUT'
        
        self.adj = {i: [] for i in range(self.num_nodes)}
        self.parents = {i: [] for i in range(self.num_nodes)}
        for src, dst in self.edges_list:
            self.adj[src].append(dst)
            self.adj[dst].append(src)
            self.parents[dst].append(src)

        self.edge_index = torch.tensor(self.edges_list, dtype=torch.long).t().contiguous()
        if self.edge_index.numel() == 0:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
        
        self.x_base = self._build_base_features()
    
    def _calculate_controllability(self):
        cc0 = {i: 1 for i in range(self.num_nodes)}
        cc1 = {i: 1 for i in range(self.num_nodes)}
        
        for _ in range(50):
            changed = False
            for i in range(self.num_nodes):
                inputs = self.gate_inputs[i]
                if not inputs:
                    continue
                
                name = self.ordered_names[i]
                g_type = self.gate_types.get(name, 'INPUT')
                
                c0s = [cc0[x] for x in inputs]
                c1s = [cc1[x] for x in inputs]
                old_c0, old_c1 = cc0[i], cc1[i]
                new_c0, new_c1 = old_c0, old_c1
                
                if g_type == 'AND':
                    new_c0, new_c1 = min(c0s) + 1, sum(c1s) + 1
                elif g_type == 'OR':
                    new_c0, new_c1 = sum(c0s) + 1, min(c1s) + 1
                elif g_type == 'NAND':
                    new_c0, new_c1 = sum(c1s) + 1, min(c0s) + 1
                elif g_type == 'NOR':
                    new_c0, new_c1 = min(c1s) + 1, sum(c0s) + 1
                elif g_type == 'NOT':
                    new_c0, new_c1 = c1s[0] + 1, c0s[0] + 1
                elif g_type == 'BUFF':
                    new_c0, new_c1 = c0s[0] + 1, c1s[0] + 1
                elif g_type == 'XOR':
                    new_c0 = min(c0s[0] + c0s[1], c1s[0] + c1s[1]) + 1
                    new_c1 = min(c0s[0] + c1s[1], c1s[0] + c0s[1]) + 1
                
                if new_c0 != old_c0 or new_c1 != old_c1:
                    cc0[i] = min(new_c0, 5000)
                    cc1[i] = min(new_c1, 5000)
                    changed = True
            
            if not changed:
                break
        
        return cc0, cc1
    
    def _calculate_observability(self):
        co = {i: float('inf') for i in range(self.num_nodes)}
        
        output_set = set(self.parser.all_outputs)
        for i, name in enumerate(self.ordered_names):
            if name in output_set:
                co[i] = 0
        
        cc0, cc1 = self._calculate_controllability()
        
        for _ in range(50):
            changed = False
            
            for i in range(self.num_nodes):
                fanout_gates = self.gate_outputs[i]
                if not fanout_gates:
                    continue
                
                min_co = co[i]
                
                for gate_idx in fanout_gates:
                    gate_name = self.ordered_names[gate_idx]
                    g_type = self.gate_types.get(gate_name, 'INPUT')
                    gate_inputs = self.gate_inputs[gate_idx]
                    
                    gate_co = co[gate_idx]
                    if gate_co == float('inf'):
                        continue
                    
                    side_input_cost = 0
                    
                    if g_type in ['AND', 'NAND']:
                        for inp_idx in gate_inputs:
                            if inp_idx != i:
                                side_input_cost += cc1[inp_idx]
                    elif g_type in ['OR', 'NOR']:
                        for inp_idx in gate_inputs:
                            if inp_idx != i:
                                side_input_cost += cc0[inp_idx]
                    elif g_type in ['NOT', 'BUFF']:
                        side_input_cost = 0
                    elif g_type == 'XOR':
                        for inp_idx in gate_inputs:
                            if inp_idx != i:
                                side_input_cost += min(cc0[inp_idx], cc1[inp_idx])
                    
                    path_co = gate_co + side_input_cost + 1
                    min_co = min(min_co, path_co)
                
                if min_co < co[i]:
                    co[i] = min(min_co, 5000)
                    changed = True
            
            if not changed:
                break
        
        for i in range(self.num_nodes):
            if co[i] == float('inf'):
                co[i] = 10000
        
        return co
    
    def _compute_depth(self, reverse=False):
        adj = {i: [] for i in range(self.num_nodes)}
        deg = {i: 0 for i in range(self.num_nodes)}
        
        for src, dst in self.edges_list:
            if reverse:
                adj[dst].append(src)
                deg[src] += 1
            else:
                adj[src].append(dst)
                deg[dst] += 1
        
        q = [i for i in range(self.num_nodes) if deg[i] == 0]
        depth = {i: 0 for i in range(self.num_nodes)}
        
        while q:
            n = q.pop(0)
            for neighbor in adj[n]:
                depth[neighbor] = max(depth[neighbor], depth[n] + 1)
                deg[neighbor] -= 1
                if deg[neighbor] == 0:
                    q.append(neighbor)
        
        max_d = max(depth.values()) if depth else 1
        return [depth[i] / max(max_d, 1) for i in range(self.num_nodes)]
    
    def _build_base_features(self):
        """
        16-dimensional features:
        [0-7]: Gate type one-hot
        [8-9]: Forward/backward depth
        [10-11]: Fault target and distance (set per-fault)
        [12-13]: SCOAP CC0, CC1
        [14]: SCOAP CO (observability)
        [15]: Is output node
        """
        types = ['INPUT', 'NAND', 'AND', 'NOR', 'OR', 'NOT', 'BUFF', 'XOR']
        t_map = {t: i for i, t in enumerate(types)}
        
        fwd = self._compute_depth(False)
        rev = self._compute_depth(True)
        cc0, cc1 = self._calculate_controllability()
        co = self._calculate_observability()
        
        x = torch.zeros((self.num_nodes, 16), dtype=torch.float)
        
        output_set = set(self.parser.all_outputs)
        
        for i, name in enumerate(self.ordered_names):
            g_type = self.gate_types.get(name, 'INPUT')
            if g_type in t_map:
                x[i, t_map[g_type]] = 1.0
            
            x[i, 8] = fwd[i]
            x[i, 9] = rev[i]
            x[i, 12] = math.log(cc0[i] + 1) / 10.0
            x[i, 13] = math.log(cc1[i] + 1) / 10.0
            x[i, 14] = math.log(co[i] + 1) / 10.0
            
            if name in output_set:
                x[i, 15] = 1.0
        
        return x
    
    def get_data_for_fault(self, fault_name):
        x = self.x_base.clone()
        
        tid = self.name_to_idx.get(fault_name)
        if tid is not None:
            x[tid, 10] = 1.0
            
            q = [tid]
            dist = {i: -1 for i in range(self.num_nodes)}
            dist[tid] = 0
            
            while q:
                n = q.pop(0)
                if dist[n] >= 10:
                    continue
                for neighbor in self.adj[n]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[n] + 1
                        q.append(neighbor)
            
            max_d = max(dist.values()) if dist else 1
            for i in range(self.num_nodes):
                if dist[i] != -1:
                    x[i, 11] = 1.0 - (dist[i] / max(max_d, 1))
        
        data = Data(x=x, edge_index=self.edge_index)
        data.node_names = self.ordered_names
        
        return data


# =============================================================================
# PART 1: DATA GENERATION WITH IMPORTANCE
# =============================================================================

def get_target_files():
    if not os.path.exists(BENCHMARK_DIR):
        return []
    return [f for f in os.listdir(BENCHMARK_DIR) if f.endswith(".bench")]


def generate_importance_aware_dataset():
    """
    Generate dataset with importance labels
    Uses 16-feature extractor with observability
    """
    print(f"--- MINING IMPORTANCE-AWARE ORACLE DATA (16 Features) ---")
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
                
                clauses = miter.build_miter(target_gate, None, 1)
                cnf = CNF()
                cnf.extend(clauses)
                
                with Glucose3(bootstrap_with=cnf) as solver:
                    if not solver.solve():
                        continue
                    
                    model = solver.get_model()
                    if not model:
                        continue
                    
                    base_conflicts = solver.accum_stats()['conflicts']
                    
                    input_importance = {}
                    input_polarity = {}  # Track which polarity reduces conflicts
                    
                    for input_name in miter.inputs:
                        var_id = miter.var_map[input_name]
                        correct_val = var_id if var_id in model else -var_id
                        wrong_val = -correct_val
                        
                        test_cnf = CNF()
                        test_cnf.extend(clauses)
                        
                        with Glucose3(bootstrap_with=test_cnf) as test_solver:
                            result = test_solver.solve(assumptions=[wrong_val])
                            
                            if result:
                                wrong_conflicts = test_solver.accum_stats()['conflicts']
                                importance = abs(wrong_conflicts - base_conflicts)
                            else:
                                importance = 10000
                        
                        input_importance[input_name] = importance
                        
                        # Store polarity: if wrong_val causes MORE conflicts, then correct_val is better
                        # Value 1 = prefer this variable TRUE, -1 = prefer this variable FALSE
                        if importance > 0:  # Variable is important
                            # The polarity that causes MORE conflicts when negated is the one to prefer
                            input_polarity[input_name] = 1 if var_id in model else -1
                        else:
                            input_polarity[input_name] = 0  # No strong preference
                    
                    max_importance = max(input_importance.values()) if input_importance else 1
                    normalized_importance = {
                        k: v / max(max_importance, 1.0) 
                        for k, v in input_importance.items()
                    }
                    
                    data = extractor.get_data_for_fault(target_gate)
                    
                    y_solution = torch.zeros(len(data.node_names), 1)
                    y_importance = torch.zeros(len(data.node_names), 1)
                    train_mask = torch.zeros(len(data.node_names), 1)
                    
                    for i, node_name in enumerate(data.node_names):
                        if node_name in input_set:
                            var_id = miter.var_map[node_name]
                            # Use polarity that reduces conflicts, not solution polarity
                            y_solution[i] = (input_polarity[node_name] + 1.0) / 2.0  # Convert -1,0,1 to 0,0.5,1
                            y_importance[i] = normalized_importance.get(node_name, 0.0)
                            train_mask[i] = 1.0
                    
                    data.y = y_solution
                    data.y_importance = y_importance
                    data.train_mask = train_mask
                    data.base_conflicts = base_conflicts
                    
                    dataset.append(data)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"--- Mining Complete. Collected {len(dataset)} samples. ---")
    torch.save(dataset, DATASET_PATH)
    print(f"--- Dataset saved to {DATASET_PATH} ---")
    return dataset


# =============================================================================
# PART 2: GNN MODEL WITH 16 FEATURES
# =============================================================================

class CircuitGNN_ImportanceAware(torch.nn.Module):
    """
    GNN for predicting variable importance in SAT solving
    """
    
    def __init__(self, num_node_features=16, num_layers=8, hidden_dim=64, dropout=0.2):
        super(CircuitGNN_ImportanceAware, self).__init__()
        
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
        
        # Only importance head - that's what works
        self.importance_head = torch.nn.Linear(32, 1)
    
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
        
        importance_scores = self.importance_head(x)
        
        return importance_scores


# =============================================================================
# PART 3: TRAINING
# =============================================================================

def train_importance_aware_model():
    """Train model with 16-feature graphs"""
    print("--- Training Importance-Aware Oracle (16 Features) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        print("Generating dataset first...")
        generate_importance_aware_dataset()
    
    dataset = torch.load(DATASET_PATH, weights_only=False)
    print(f"Loaded {len(dataset)} samples")
    
    split = int(len(dataset) * 0.8)
    train_data, val_data = dataset[:split], dataset[split:]
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    model = CircuitGNN_ImportanceAware(
        num_node_features=16,
        num_layers=8,
        dropout=0.2
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    importance_criterion = nn.MSELoss(reduction='none')
    
    for epoch in range(EPOCHS):
        model.train()
        total_importance_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            importance_preds = model(batch)
            
            importance_loss_raw = importance_criterion(importance_preds, batch.y_importance)
            importance_loss_masked = (importance_loss_raw * batch.train_mask).sum() / batch.train_mask.sum().clamp(min=1)
            
            importance_loss_masked.backward()
            optimizer.step()
            
            total_importance_loss += importance_loss_masked.item()
        
        avg_importance = total_importance_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_importance_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                importance_preds = model(batch)
                
                importance_loss_raw = importance_criterion(importance_preds, batch.y_importance)
                importance_loss_masked = (importance_loss_raw * batch.train_mask).sum() / batch.train_mask.sum().clamp(min=1)
                
                val_importance_loss += importance_loss_masked.item()
        
        avg_val_importance = val_importance_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train - Importance: {avg_importance:.5f}")
        print(f"  Val   - Importance: {avg_val_importance:.5f}")
        print(f"  Sample predictions - Importance range: [{importance_preds.min():.3f}, {importance_preds.max():.3f}]")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n--- Training Complete. Model saved to {MODEL_PATH} ---")


# =============================================================================
# PART 4: BENCHMARKING
# =============================================================================

def run_importance_guided_benchmark():
    """Benchmark with importance-guided SAT solving"""
    print(f"--- BENCHMARKING WITH IMPORTANCE-GUIDED HINTS (16 Features) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CircuitGNN_ImportanceAware(num_node_features=16, num_layers=8).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Train first!")
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    results = []
    files = get_target_files()
    
    for filename in files:
        filepath = os.path.join(BENCHMARK_DIR, filename)
        print(f"\nProcessing {filename}...")
        
        try:
            miter = WireFaultMiter(filepath)
            if not miter.gates:
                continue
            
            extractor = FastGraphExtractor(filepath, miter.var_map)
            input_names = set(miter.inputs)
            
            for i in range(20):
                target_gate = random.choice(miter.gates)[0]
                
                clauses = miter.build_miter(target_gate, None, 1)
                cnf = CNF()
                cnf.extend(clauses)
                
                t_std = time.time()
                with Glucose3(bootstrap_with=cnf, incr=True) as solver:
                    solver.solve()
                    std_conflicts = solver.accum_stats()['conflicts']
                std_time = time.time() - t_std
                
                t_gnn = time.time()
                data = extractor.get_data_for_fault(target_gate)
                if data is None:
                    continue
                data = data.to(device)
                
                with torch.no_grad():
                    importance_scores = model(data)
                
                # Get TOP_K most important input variables
                input_importance_list = []
                for idx, name in enumerate(data.node_names):
                    if name in input_names:
                        importance = importance_scores[idx].item()
                        var_id = miter.var_map.get(name)
                        if var_id:
                            input_importance_list.append((var_id, importance))
                
                # Sort by importance and get top K
                input_importance_list.sort(key=lambda x: x[1], reverse=True)
                TOP_K = 5
                top_k_vars = [h[0] for h in input_importance_list[:TOP_K]]
                
                print(f"   Top {TOP_K} important variables: {top_k_vars}")
                print(f"   Importance scores: {[f'{h[1]:.4f}' for h in input_importance_list[:TOP_K]]}")
                
                # Soft phase hints only - allows full backtracking
                with Glucose3(bootstrap_with=cnf, incr=True) as solver:
                    solver.set_phases(top_k_vars)  # Soft preference only
                    result = solver.solve()         # Solver can backtrack freely
                    gnn_conflicts = solver.accum_stats()['conflicts']
                
                gnn_time = time.time() - t_gnn
                
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
    
    if results:
        with open("importance_guided_results_16feat.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to importance_guided_results_16feat.csv")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python importance_pipeline_16feat.py generate  # Generate dataset")
        print("  python importance_pipeline_16feat.py train     # Train model")
        print("  python importance_pipeline_16feat.py benchmark # Run benchmark")
        print("\nFeatures: 16-dimensional with SCOAP observability")
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