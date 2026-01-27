"""
Data Generation with Complete ATPG Cone Extraction
Includes: fan-in + fault + fan-out + side inputs
"""

import os
import sys
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
SAMPLES_PER_FILE = 20
MAX_PROBES = 50
CONFLICT_BUDGET = 5000
SKIP_SIZE_THRESHOLD = 10000
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
# DATA GENERATION WITH COMPLETE ATPG CONES
# =============================================================================

def process_single_circuit(filename):
    """Worker function with COMPLETE ATPG cone extraction."""
    set_global_seed(SEED + hash(filename) % 10000)
    
    filepath = os.path.join(GENERATE_TRAIN_DATA_DIR, filename)
    local_dataset = []
    
    try:
        miter = WireFaultMiter(filepath)
        num_gates = len(miter.gates)
        
        if num_gates == 0:
            return []
        
        if num_gates > SKIP_SIZE_THRESHOLD:
            print(f"[{filename}] SKIP: {num_gates} gates (too large)", flush=True)
            return []
        
        # Adaptive sampling
        if num_gates > 5000:
            samples = 5
            max_probes = 30
        elif num_gates > 1000:
            samples = 10
            max_probes = 50
        else:
            samples = SAMPLES_PER_FILE
            max_probes = MAX_PROBES
        
        extractor = VectorizedGraphExtractor(filepath, var_map=miter.var_map, device='cpu')
        
        print(f"[{filename}] Processing {samples} faults with COMPLETE ATPG cones...", flush=True)
        
        # Sample faults
        if len(miter.gates) <= samples:
            sampled_faults = miter.gates
        else:
            sampled_faults = random.sample(miter.gates, samples)
        
        successful = 0
        
        for target_gate, _, _ in sampled_faults:
            fault_type = 1  # SA1
            
            # Step 1: Find reachable outputs
            reachable_outputs = miter.get_reachable_outputs(target_gate)
            if not reachable_outputs:
                continue
            
            # Step 2: Select best output (smallest cone estimate)
            best_output = None
            best_estimate = float('inf')
            
            for output in reachable_outputs[:min(20, len(reachable_outputs))]:
                # Quick estimate: distance-based heuristic
                estimate = abs(hash(output)) % 1000  # Simple heuristic
                if estimate < best_estimate:
                    best_estimate = estimate
                    best_output = output
            
            if best_output is None:
                best_output = reachable_outputs[0]
            
            # Step 3: Extract COMPLETE ATPG CONE
            complete_cone = miter.get_complete_atpg_cone(target_gate, best_output)
            
            if not complete_cone:
                continue
            
            # Step 4: Temporarily swap gates to solve cone only
            orig_gates = miter.gates
            miter.gates = complete_cone
            
            clauses = miter.build_miter(target_gate, fault_type, 1)
            miter.gates = orig_gates  # Restore immediately
            
            # Step 5: Solve the miter
            with Glucose3(bootstrap_with=clauses) as solver:
                solver.conf_budget(CONFLICT_BUDGET)
                if not solver.solve():
                    continue
                
                assignment = solver.get_model()
                if not assignment:
                    continue
            
            # Step 6: Extract cone inputs for probing
            cone_inputs = miter.get_cone_inputs(complete_cone)
            
            if not cone_inputs:
                continue
            
            probe_list = list(cone_inputs)
            if len(probe_list) > max_probes:
                probe_list = random.sample(probe_list, max_probes)
            
            # Step 7: Probe inputs to measure importance
            with Glucose3(bootstrap_with=clauses) as probe:
                base_conflicts = probe.accum_stats()['conflicts']
                input_importance = {}
                input_polarity = {}
                
                for inp in probe_list:
                    if inp not in miter.var_map:
                        continue
                    
                    var_id = miter.var_map[inp]
                    correct_val = var_id if var_id in assignment else -var_id
                    
                    # Try flipping the input
                    probe.conf_budget(500)
                    result = probe.solve(assumptions=[-correct_val])
                    
                    new_conflicts = probe.accum_stats()['conflicts']
                    delta = new_conflicts - base_conflicts
                    base_conflicts = new_conflicts
                    
                    if result:
                        input_importance[inp] = delta
                    else:
                        input_importance[inp] = 5000  # UNSAT = Critical
                    
                    input_polarity[inp] = 1.0 if var_id in assignment else 0.0
            
            if not input_importance:
                continue
            
            # Step 8: Build training sample
            data = extractor.get_data_for_fault(target_gate, fault_type=fault_type)
            max_imp = max(input_importance.values())
            
            y_polarity = torch.zeros(len(data.node_names), 1)
            y_importance = torch.zeros(len(data.node_names), 1)
            train_mask = torch.zeros(len(data.node_names), 1)
            
            for k, node_name in enumerate(data.node_names):
                if node_name in input_importance:
                    y_polarity[k] = input_polarity[node_name]
                    y_importance[k] = input_importance[node_name] / max(max_imp, 1.0)
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
            if f.endswith(".bench") or f.endswith(".v"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, DIR)
                file_list.append(rel_path)
    
    return sorted(file_list, key=lambda x: os.path.getsize(os.path.join(DIR, x)))

def generate_dataset():
    """Main dataset generation function"""
    print("=" * 80)
    print("DATA GENERATION (Complete ATPG Cones)")
    print("=" * 80)
    print(f"Directory: {GENERATE_TRAIN_DATA_DIR}")
    print(f"Max probes per fault: {MAX_PROBES}")
    print(f"Samples per file: {SAMPLES_PER_FILE}")
    print(f"Workers: {NUM_WORKERS}")
    print("=" * 80)
    
    files = get_target_files(GENERATE_TRAIN_DATA_DIR)
    
    if not files:
        print(f"ERROR: No .bench or .v files found in {GENERATE_TRAIN_DATA_DIR}")
        return
    
    print(f"Found {len(files)} benchmark files")
    
    dataset = []
    
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass
    
    start_time = time.time()
    
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_circuit, files),
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