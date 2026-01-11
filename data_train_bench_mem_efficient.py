"""
Memory-efficient pipeline that saves only labels and reconstructs graphs during training
"""

import os
import sys
import time
import csv
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import torch.optim as optim
import random
from pysat.solvers import Glucose3
from pysat.formula import CNF
from tqdm import tqdm
from torch_geometric.data import Dataset
from WireFaultMiter import WireFaultMiter
from neuro_utils import FastGraphExtractor

# Configs
BENCH_DIR = "../synthetic_bench"
DATASET_PATH = "dataset_labels_only.pt"  # Much smaller file!
SAMPLES_PER_FILE = 50
MODEL_PATH = "gnn_model_importance_aware.pth"
EPOCHS = 20
BATCH_SIZE = 32
BENCHMARK_DIR = "../hdl-benchmarks/iscas89/bench/"

# =============================================================================
# PART 1: LIGHTWEIGHT DATA GENERATION (Labels Only)
# =============================================================================

def generate_labels_only_dataset():
    """
    Save only the essential information needed to reconstruct training samples:
    - Circuit filename
    - Fault wire name
    - Input labels (solution values)
    - Input importance scores
    - Metadata (conflicts, etc.)
    
    This is MUCH smaller than saving full graphs!
    """
    print(f"--- MINING LABELS (Lightweight Mode) ---")
    label_records = []
    
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
            
            input_names = list(miter.inputs)
            
            for _ in range(SAMPLES_PER_FILE):
                target_gate = random.choice(miter.gates)[0]
                
                # === STEP 1: Get correct solution ===
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
                    
                    # === STEP 2: Measure importance per input ===
                    input_importance = {}
                    
                    for input_name in input_names:
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
                    
                    # Normalize
                    max_importance = max(input_importance.values()) if input_importance else 1
                    normalized_importance = {
                        k: v / max(max_importance, 1.0) 
                        for k, v in input_importance.items()
                    }
                    
                    # === STEP 3: Store ONLY the labels (not the graph!) ===
                    # Extract solution values
                    solution_dict = {}
                    for input_name in input_names:
                        var_id = miter.var_map[input_name]
                        solution_dict[input_name] = 1.0 if var_id in model else 0.0
                    
                    # Create lightweight record
                    record = {
                        'circuit_file': filename,           # e.g., "s298.bench"
                        'fault_wire': target_gate,          # e.g., "wire_42"
                        'solution': solution_dict,          # {input_name: 0 or 1}
                        'importance': normalized_importance,# {input_name: 0.0-1.0}
                        'base_conflicts': base_conflicts    # For analysis
                    }
                    
                    label_records.append(record)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"--- Mining Complete. Collected {len(label_records)} label records. ---")
    torch.save(label_records, DATASET_PATH)
    print(f"--- Labels saved to {DATASET_PATH} ---")
    
    # Report size savings
    import sys
    size_mb = sys.getsizeof(label_records) / (1024 * 1024)
    print(f"Dataset size: ~{size_mb:.2f} MB (much smaller than full graphs!)")
    
    return label_records


# =============================================================================
# PART 2: LAZY-LOADING DATASET CLASS
# =============================================================================

class LazyCircuitDataset(Dataset):
    """
    PyTorch Geometric Dataset that reconstructs graphs on-the-fly during training.
    
    This saves memory by not storing full graphs, only labels.
    Graphs are built as needed from the bench files.
    """
    
    def __init__(self, label_records, bench_dir, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.label_records = label_records
        self.bench_dir = bench_dir
        
        # Cache extractors to avoid re-parsing the same file
        self.extractor_cache = {}
    
    def len(self):
        return len(self.label_records)
    
    def get(self, idx):
        """
        Build graph on-the-fly from stored labels.
        This is called by the DataLoader during training.
        """
        record = self.label_records[idx]
        
        # Get or create extractor for this circuit
        circuit_file = record['circuit_file']
        if circuit_file not in self.extractor_cache:
            filepath = os.path.join(self.bench_dir, circuit_file)
            
            # Parse the bench file and build extractor
            miter = WireFaultMiter(filepath)
            extractor = FastGraphExtractor(filepath, miter.var_map)
            
            self.extractor_cache[circuit_file] = {
                'extractor': extractor,
                'miter': miter
            }
        
        cache_entry = self.extractor_cache[circuit_file]
        extractor = cache_entry['extractor']
        miter = cache_entry['miter']
        
        # Build graph structure for this fault
        fault_wire = record['fault_wire']
        data = extractor.get_data_for_fault(fault_wire)
        
        # Attach labels from the saved record
        input_set = set(miter.inputs)
        y_solution = torch.zeros(len(data.node_names), 1)
        y_importance = torch.zeros(len(data.node_names), 1)
        train_mask = torch.zeros(len(data.node_names), 1)
        
        for i, node_name in enumerate(data.node_names):
            if node_name in input_set:
                # Get labels from saved record
                y_solution[i] = record['solution'].get(node_name, 0.0)
                y_importance[i] = record['importance'].get(node_name, 0.0)
                train_mask[i] = 1.0
        
        # Attach to data object
        data.y = y_solution
        data.y_importance = y_importance
        data.train_mask = train_mask
        data.base_conflicts = record['base_conflicts']
        
        return data


# =============================================================================
# PART 3: GNN MODEL (Same as before)
# =============================================================================

class CircuitGNN_ImportanceAware(torch.nn.Module):
    def __init__(self, num_node_features=14, num_layers=8, hidden_dim=64, dropout=0.2):
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
        
        self.value_head = torch.nn.Linear(32, 1)
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
        
        value_logits = self.value_head(x)
        importance_scores = self.importance_head(x)
        
        return value_logits, importance_scores


# =============================================================================
# PART 4: TRAINING WITH LAZY LOADING
# =============================================================================

def train_with_lazy_loading():
    """
    Train using lazy-loaded dataset.
    Graphs are reconstructed on-the-fly from labels.
    """
    print("--- Training with Lazy Loading ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Labels not found at {DATASET_PATH}")
        print("Generating labels first...")
        generate_labels_only_dataset()
    
    # Load label records (lightweight!)
    label_records = torch.load(DATASET_PATH, weights_only=False)
    print(f"Loaded {len(label_records)} label records")
    
    # Create lazy dataset
    dataset = LazyCircuitDataset(label_records, BENCH_DIR)
    
    # Train/Val split
    split = int(len(dataset) * 0.8)
    train_indices = list(range(split))
    val_indices = list(range(split, len(dataset)))
    
    # Create datasets for each split
    train_dataset = LazyCircuitDataset(
        [label_records[i] for i in train_indices],
        BENCH_DIR
    )
    val_dataset = LazyCircuitDataset(
        [label_records[i] for i in val_indices],
        BENCH_DIR
    )
    
    # Create data loaders (graphs built on-demand)
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    model = CircuitGNN_ImportanceAware(
        num_node_features=14, 
        num_layers=8, 
        dropout=0.2
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
            
            value_logits, importance_preds = model(batch)
            
            value_loss_raw = value_criterion(value_logits, batch.y)
            value_loss_masked = (value_loss_raw * batch.train_mask).sum() / batch.train_mask.sum().clamp(min=1)
            
            importance_loss_raw = importance_criterion(importance_preds, batch.y_importance)
            importance_loss_masked = (importance_loss_raw * batch.train_mask).sum() / batch.train_mask.sum().clamp(min=1)
            
            combined_loss = 0.6 * value_loss_masked + 0.4 * importance_loss_masked
            
            combined_loss.backward()
            optimizer.step()
            
            total_value_loss += value_loss_masked.item()
            total_importance_loss += importance_loss_masked.item()
            total_combined_loss += combined_loss.item()
        
        avg_value = total_value_loss / len(train_loader)
        avg_importance = total_importance_loss / len(train_loader)
        avg_combined = total_combined_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Value Loss: {avg_value:.5f}")
        print(f"  Importance Loss: {avg_importance:.5f}")
        print(f"  Combined Loss: {avg_combined:.5f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n--- Training Complete. Model saved to {MODEL_PATH} ---")


# =============================================================================
# PART 5: BENCHMARK (Same as before)
# =============================================================================

def get_target_files():
    if not os.path.exists(BENCHMARK_DIR):
        return []
    return [f for f in os.listdir(BENCHMARK_DIR) if f.endswith(".bench")]


def run_importance_guided_benchmark():
    """Use trained model for SAT guidance"""
    print(f"--- BENCHMARKING WITH IMPORTANCE-GUIDED HINTS ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
                
                # Baseline
                clauses = miter.build_miter(target_gate, None, 1)
                cnf = CNF()
                cnf.extend(clauses)
                
                t_std = time.time()
                with Glucose3(bootstrap_with=cnf) as solver:
                    solver.solve()
                    std_conflicts = solver.accum_stats()['conflicts']
                std_time = time.time() - t_std
                
                # GNN-Guided
                t_gnn = time.time()
                data = extractor.get_data_for_fault(target_gate)
                if data is None:
                    continue
                data = data.to(device)
                
                with torch.no_grad():
                    value_logits, importance_scores = model(data)
                    value_probs = torch.sigmoid(value_logits)
                
                hints = []
                for idx, name in enumerate(data.node_names):
                    if name in input_names:
                        prob = value_probs[idx].item()
                        value = 1 if prob > 0.5 else -1
                        importance = importance_scores[idx].item()
                        
                        var_id = miter.var_map.get(name)
                        if var_id:
                            hints.append((var_id, value, importance))
                
                hints.sort(key=lambda x: x[2], reverse=True)
                
                TOP_K = 5
                top_assumptions = [h[0] * h[1] for h in hints[:TOP_K]]
                
                gnn_conflicts = 0
                with Glucose3(bootstrap_with=cnf) as solver:
                    if solver.solve(assumptions=top_assumptions):
                        gnn_conflicts = solver.accum_stats()['conflicts']
                    else:
                        solver.solve()
                        gnn_conflicts = solver.accum_stats()['conflicts'] + 1000
                
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
        print("  python lazy_pipeline.py generate  # Generate labels only")
        print("  python lazy_pipeline.py train     # Train with lazy loading")
        print("  python lazy_pipeline.py benchmark # Run benchmark")
    else:
        command = sys.argv[1]
        
        if command == "generate":
            generate_labels_only_dataset()
        elif command == "train":
            train_with_lazy_loading()
        elif command == "benchmark":
            run_importance_guided_benchmark()
        else:
            print(f"Unknown command: {command}")