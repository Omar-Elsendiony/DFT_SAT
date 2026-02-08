"""
Data Generation with Complete ATPG Cone Extraction (Polarity Optimized)
Fast version: Extracts polarity directly from SAT model without expensive probing.
"""

import os
import sys
# Ensure pysat is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pysat'))

import time
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from pysat.solvers import Glucose3
from tqdm import tqdm
from WireFaultMiter import WireFaultMiter
from neuro_utils import VectorizedGraphExtractor

# =============================================================================
# CONFIGS
# =============================================================================
GENERATE_TRAIN_DATA_DIR = "../I99T"
DATASET_PATH = "dataset_complete_atpg_17feat.pt"
SAMPLES_PER_FILE = 30
# Max input nodes to include in the training mask per fault
MAX_INPUTS_PER_FAULT = 200 
CONFLICT_BUDGET = 5000
SKIP_SIZE_THRESHOLD = 100000
NUM_WORKERS = 8
SEED = 42

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# =============================================================================
# FAST CONE SIZE ESTIMATION
# =============================================================================

def estimate_cone_size_fast(miter, target_gate, target_output):
    """BFS depth-limited traversal to find the most 'local' output for a fault."""
    fanout_size = 0
    visited = set()
    queue = [(target_gate, 0)]
    
    while queue and len(visited) < 100:
        node, depth = queue.pop(0)
        if node in visited or depth > 10: continue
        visited.add(node)
        if node in miter.parser.gate_dict: fanout_size += 1
        if node == target_output: break
        
        fanout = miter.parser.get_fanout(node)
        for next_gate in fanout:
            if next_gate not in visited:
                queue.append((next_gate, depth + 1))
    
    return fanout_size

# =============================================================================
# WORKER FUNCTION
# =============================================================================

def process_single_circuit(filename):
    set_global_seed(SEED + hash(filename) % 10000)
    filepath = os.path.join(GENERATE_TRAIN_DATA_DIR, filename)
    local_dataset = []
    
    try:
        miter = WireFaultMiter(filepath)
        if len(miter.gates) == 0 or len(miter.gates) > SKIP_SIZE_THRESHOLD:
            return []
        
        # Adaptive sampling based on circuit size
        samples = 5 if len(miter.gates) > 5000 else SAMPLES_PER_FILE
        extractor = VectorizedGraphExtractor(filepath, var_map=miter.var_map, device='cpu')
        
        sampled_faults = random.sample(miter.gates, min(len(miter.gates), samples))
        successful = 0
        
        for target_gate, _, _ in sampled_faults:
            fault_type = 1  # Stuck-at-1
            
            reachable_outputs = miter.get_reachable_outputs(target_gate)
            if not reachable_outputs: continue
            
            # Select best output (closest/smallest cone)
            best_output = min(random.sample(reachable_outputs, min(5, len(reachable_outputs))), 
                              key=lambda out: estimate_cone_size_fast(miter, target_gate, out))
            
            complete_cone = miter.get_complete_atpg_cone(target_gate, best_output)
            if not complete_cone: continue
            
            # Solve for polarity
            clauses = miter.build_miter(target_gate, fault_type, 1)
            with Glucose3(bootstrap_with=clauses) as solver:
                solver.conf_budget(CONFLICT_BUDGET)
                if not solver.solve(): continue
                
                assignment = set(solver.get_model())
            
            # Step 7: Optimized Polarity Extraction
            cone_inputs = miter.get_cone_inputs(complete_cone)
            input_polarity = {}
            for inp in cone_inputs:
                if inp in miter.var_map:
                    var_id = miter.var_map[inp]
                    input_polarity[inp] = 1.0 if var_id in assignment else 0.0

            # Step 8: Build GNN Data Object
            data = extractor.get_data_for_fault(target_gate, fault_type=fault_type)
            y_polarity = torch.zeros(len(data.node_names), 1)
            train_mask = torch.zeros(len(data.node_names), 1)
            
            # Map solver assignment back to graph nodes
            for k, node_name in enumerate(data.node_names):
                if node_name in input_polarity:
                    y_polarity[k] = input_polarity[node_name]
                    train_mask[k] = 1.0
            
            data.y_polarity = y_polarity
            data.train_mask = train_mask
            # Dummy importance to maintain structure if needed for other legacy scripts
            data.y_importance = torch.zeros_like(y_polarity) 
            
            local_dataset.append(data)
            successful += 1
            
    except Exception as e:
        print(f"[{filename}] Error: {e}")
    
    return local_dataset

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def generate_dataset():
    print("=" * 80)
    print("FAST DATA GENERATION (Polarity Extraction Mode)")
    print("=" * 80)
    
    # Recursive search for .bench or .v files
    files = []
    for root, _, filenames in os.walk(GENERATE_TRAIN_DATA_DIR):
        for f in filenames:
            if f.endswith(".bench") or f.endswith(".v"):
                files.append(os.path.relpath(os.path.join(root, f), GENERATE_TRAIN_DATA_DIR))
    
    files = sorted(files, key=lambda x: os.path.getsize(os.path.join(GENERATE_TRAIN_DATA_DIR, x)))
    print(f"Found {len(files)} benchmark files. Using {NUM_WORKERS} workers.")
    
    dataset = []
    start_time = time.time()
    
    try:
        mp.set_start_method('spawn', force=True)
    except: pass
    
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_circuit, files), total=len(files)))
        for res in results: dataset.extend(res)
    
    elapsed = time.time() - start_time
    print(f"\nDone! Generated {len(dataset)} samples in {elapsed/60:.2f} mins.")
    
    if dataset:
        torch.save(dataset, DATASET_PATH)
        print(f"Dataset saved to: {DATASET_PATH}")

if __name__ == "__main__":
    generate_dataset()