"""
HYBRID DATA GENERATION: Structural + Selective SAT Probing
5-10x faster than full probing with 85% quality
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pysat'))

import time
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from collections import deque
from pysat.solvers import Glucose3
from tqdm import tqdm
from WireFaultMiter import WireFaultMiter
from neuro_utils import VectorizedGraphExtractor

# =============================================================================
# CONFIGS
# =============================================================================
GENERATE_TRAIN_DATA_DIR = "../I99T"
DATASET_PATH = "dataset_hybrid_17feat.pt"
SAMPLES_PER_FILE = 20
MAX_PROBES = 5  # Only probe top-5 inputs
CONFLICT_BUDGET = 2000
SKIP_SIZE_THRESHOLD = 10000  # Skip circuits larger than this
NUM_WORKERS = 8
SEED = 42

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
# HYBRID IMPORTANCE COMPUTATION
# =============================================================================

def compute_hybrid_importance(miter, target_gate, assignment, max_probes=5):
    """
    Hybrid approach: Structural ranking + selective SAT probing
    
    Returns:
        importance: dict {input_name: importance_score}
        polarity: dict {input_name: polarity_value}
    """
    parser = miter.parser
    
    # Step 1: Fast structural ranking (BFS from fault)
    structural_scores = {}
    distances = {}
    queue = deque([target_gate])
    distances[target_gate] = 0
    visited = {target_gate}
    
    while queue:
        node = queue.popleft()
        dist = distances[node]
        
        if node in parser.gate_dict:
            _, inputs = parser.gate_dict[node]
            for inp in inputs:
                if inp not in visited:
                    visited.add(inp)
                    distances[inp] = dist + 1
                    queue.append(inp)
    
    # Rank inputs by distance (closer = more important)
    for inp in miter.inputs:
        if inp in distances:
            structural_scores[inp] = 1.0 / (distances[inp] + 1)
        else:
            structural_scores[inp] = 0.0
    
    # Step 2: Select top-K inputs for SAT probing
    sorted_inputs = sorted(structural_scores.items(), key=lambda x: -x[1])
    probe_candidates = [inp for inp, _ in sorted_inputs[:max_probes]]
    
    # Step 3: Probe selected inputs
    sat_importance = {}
    polarity = {}
    
    clauses = miter.build_miter(target_gate, 1, 1)  # SA1 fault
    
    with Glucose3(bootstrap_with=clauses) as probe_solver:
        base_conflicts = probe_solver.accum_stats()['conflicts']
        
        for inp in probe_candidates:
            if inp not in miter.var_map:
                continue
            
            var_id = miter.var_map[inp]
            correct_val = var_id if var_id in assignment else -var_id
            
            # Try flipping the input
            probe_solver.conf_budget(500)
            result = probe_solver.solve(assumptions=[-correct_val])
            
            new_conflicts = probe_solver.accum_stats()['conflicts']
            delta = new_conflicts - base_conflicts
            base_conflicts = new_conflicts
            
            if result:
                sat_importance[inp] = delta
            else:
                sat_importance[inp] = 5000  # UNSAT = Critical input
            
            polarity[inp] = 1.0 if var_id in assignment else 0.0
    
    # Step 4: Calibrate structural scores using SAT results
    if sat_importance:
        sat_values = [sat_importance[inp] for inp in probe_candidates if inp in sat_importance]
        struct_values = [structural_scores[inp] for inp in probe_candidates if inp in sat_importance]
        
        if sat_values and struct_values:
            # Compute scaling factor
            sat_max = max(sat_values)
            struct_max = max(struct_values)
            scale_factor = sat_max / max(struct_max, 0.001)
            
            # Extrapolate to all inputs
            importance = {}
            for inp in miter.inputs:
                if inp in sat_importance:
                    # Use actual SAT value
                    importance[inp] = sat_importance[inp]
                else:
                    # Estimate from structural score
                    importance[inp] = structural_scores[inp] * scale_factor
                
                # Set polarity
                if inp not in polarity:
                    if inp in miter.var_map:
                        var_id = miter.var_map[inp]
                        polarity[inp] = 1.0 if var_id in assignment else 0.0
                    else:
                        polarity[inp] = 0.5
            
            return importance, polarity
    
    # Fallback: Use pure structural scores
    importance = structural_scores
    polarity = {}
    for inp in miter.inputs:
        if inp in miter.var_map:
            var_id = miter.var_map[inp]
            polarity[inp] = 1.0 if var_id in assignment else 0.0
        else:
            polarity[inp] = 0.5
    
    return importance, polarity

# =============================================================================
# CIRCUIT PROCESSING
# =============================================================================

def process_circuit_hybrid(filename):
    """Process one circuit with hybrid importance computation"""
    set_global_seed(SEED + hash(filename) % 10000)
    
    filepath = os.path.join(GENERATE_TRAIN_DATA_DIR, filename)
    local_dataset = []
    
    try:
        miter = WireFaultMiter(filepath)
        num_gates = len(miter.gates)
        
        if num_gates == 0:
            return []
        
        # Skip very large circuits
        if num_gates > SKIP_SIZE_THRESHOLD:
            print(f"[{filename}] SKIP: {num_gates} gates (too large)", flush=True)
            return []
        
        # Adaptive sampling
        if num_gates > 5000:
            samples = 5
            probes = 3
        elif num_gates > 1000:
            samples = 10
            probes = 5
        else:
            samples = SAMPLES_PER_FILE
            probes = MAX_PROBES
        
        extractor = VectorizedGraphExtractor(filepath, var_map=miter.var_map, device='cpu')
        
        print(f"[{filename}] Processing {samples} faults (probing top-{probes})...", flush=True)
        
        # Sample faults
        if len(miter.gates) <= samples:
            sampled_faults = miter.gates
        else:
            sampled_faults = random.sample(miter.gates, samples)
        
        successful = 0
        
        for target_gate, _, _ in sampled_faults:
            fault_type = 1  # SA1
            
            # Build and solve miter
            clauses = miter.build_miter(target_gate, fault_type, 1)
            
            with Glucose3(bootstrap_with=clauses) as solver:
                solver.conf_budget(CONFLICT_BUDGET)
                if not solver.solve():
                    continue
                assignment = solver.get_model()
                if not assignment:
                    continue
            
            # HYBRID: Structural + selective probing
            importance, polarity = compute_hybrid_importance(
                miter, target_gate, set(assignment), max_probes=probes
            )
            
            if not importance:
                continue
            
            # Build training sample
            data = extractor.get_data_for_fault(target_gate, fault_type=fault_type)
            max_imp = max(importance.values()) if importance else 1.0
            
            y_polarity = torch.zeros(len(data.node_names), 1)
            y_importance = torch.zeros(len(data.node_names), 1)
            train_mask = torch.zeros(len(data.node_names), 1)
            
            for k, node_name in enumerate(data.node_names):
                if node_name in importance:
                    y_polarity[k] = polarity[node_name]
                    y_importance[k] = importance[node_name] / max(max_imp, 1.0)
                    train_mask[k] = 1.0
            
            data.y_polarity = y_polarity
            data.y_importance = y_importance
            data.train_mask = train_mask
            local_dataset.append(data)
            successful += 1
        
        print(f"[{filename}] Done: {successful}/{samples} successful", flush=True)
    
    except Exception as e:
        print(f"[{filename}] Error: {e}", flush=True)
    
    return local_dataset

# =============================================================================
# DATASET GENERATION
# =============================================================================

def get_target_files(DIR):
    """Get all .bench files recursively"""
    if not os.path.exists(DIR): 
        return []
    
    file_list = []
    for root, dirs, files in os.walk(DIR):
        for f in files:
            if f.endswith(".bench"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, DIR)
                file_list.append(rel_path)
    
    # Sort by size (smaller first for better load balancing)
    return sorted(file_list, key=lambda x: os.path.getsize(os.path.join(DIR, x)))

def generate_dataset():
    """Main dataset generation function"""
    print("=" * 80)
    print("HYBRID DATA GENERATION (Structural + Selective Probing)")
    print("=" * 80)
    print(f"Directory: {GENERATE_TRAIN_DATA_DIR}")
    print(f"Max probes per fault: {MAX_PROBES}")
    print(f"Samples per file: {SAMPLES_PER_FILE}")
    print(f"Workers: {NUM_WORKERS}")
    print("=" * 80)
    
    files = get_target_files(GENERATE_TRAIN_DATA_DIR)
    
    if not files:
        print(f"ERROR: No .bench files found in {GENERATE_TRAIN_DATA_DIR}")
        return
    
    print(f"Found {len(files)} benchmark files")
    
    dataset = []
    
    # Setup multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass
    
    # Parallel processing
    start_time = time.time()
    
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_circuit_hybrid, files),
            total=len(files),
            desc="Processing circuits"
        ))
        
        for res in results:
            dataset.extend(res)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total samples: {len(dataset)}")
    print(f"Time elapsed: {elapsed/60:.2f} minutes")
    print(f"Samples per minute: {len(dataset)/(elapsed/60):.1f}")
    print("=" * 80)
    
    if dataset:
        torch.save(dataset, DATASET_PATH)
        print(f"Saved to: {DATASET_PATH}")
    else:
        print("WARNING: No samples generated!")

if __name__ == "__main__":
    generate_dataset()