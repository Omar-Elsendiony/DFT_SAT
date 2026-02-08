"""
Benchmark GNN-Guided SAT Solving for Circuit Testability

Note: Only uses POLARITY predictions (importance head removed from usage)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pysat'))

import time
import csv
import random
import numpy as np
import torch
from pysat.solvers import Minisat22
from WireFaultMiter import WireFaultMiter
from neuro_utils import VectorizedGraphExtractor
from train_model import CircuitGNN_DualTask

# =============================================================================
# CONFIGS
# =============================================================================
BENCHMARK_DIR = "../hdl-benchmarks/iscas85/bench/"
MODEL_PATH = "gnn_model_dual_task_17feat_improved.pth"
RESULTS_PATH = "results_gnn_polarity.csv"
NUM_FAULTS_PER_CIRCUIT = 10
SEED = 42

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_global_seed(SEED)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_benchmark_files(benchmark_dir):
    """Get all .bench files from benchmark directory"""
    if not os.path.exists(benchmark_dir):
        return []
    
    files = []
    for root, dirs, filenames in os.walk(benchmark_dir):
        for f in filenames:
            if f.endswith(".bench"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, benchmark_dir)
                files.append(rel_path)
    
    return sorted(files)

# =============================================================================
# BENCHMARKING
# =============================================================================

def run_benchmark():
    """Main benchmarking function"""
    print("=" * 80)
    print("GNN-GUIDED SAT SOLVING BENCHMARK (Polarity Only)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please train the model first!")
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    model = CircuitGNN_DualTask(num_node_features=17).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # Get benchmark files
    files = get_benchmark_files(BENCHMARK_DIR)
    if not files:
        print(f"ERROR: No benchmark files found in {BENCHMARK_DIR}")
        return
    
    print(f"Found {len(files)} benchmark circuits")
    print("=" * 80)
    
    results = []
    
    for filename in files:
        filepath = os.path.join(BENCHMARK_DIR, filename)
        print(f"\nProcessing {filename}...")
        
        try:
            miter = WireFaultMiter(filepath)
            num_gates = len(miter.gates)
            
            if num_gates == 0:
                print(f"  Skipping (no gates)")
                continue
            
            if num_gates > 10000:
                print(f"  Skipping (too large: {num_gates} gates)")
                continue
            
            print(f"  Gates: {num_gates}, Inputs: {len(miter.inputs)}")
            
            # Create extractor
            extractor = VectorizedGraphExtractor(filepath, var_map=miter.var_map, device=device.type)
            
            # Sample faults
            all_gates = sorted(miter.gates, key=lambda x: x[0])
            num_tests = min(NUM_FAULTS_PER_CIRCUIT, len(all_gates))
            
            for i in range(num_tests):
                target_gate = random.choice(all_gates)[0]
                fault_type = 1  # SA1
                
                # === BASELINE: Standard SAT solving ===
                clauses_std = miter.build_miter(target_gate, fault_type, 1)
                
                t_std_start = time.time()
                with Minisat22(bootstrap_with=clauses_std) as solver:
                    std_result = solver.solve()
                    std_conflicts = solver.accum_stats()['conflicts']
                std_time = time.time() - t_std_start
                
                # === GNN-GUIDED: With polarity hints ===
                t_gnn_start = time.time()
                
                # 1. Get GNN predictions
                data = extractor.get_data_for_fault(target_gate, fault_type=fault_type).to(device)
                
                with torch.no_grad():
                    _, pol_scores = model(data)  # Only use polarity
                
                # 2. Build phase hints (polarity only)
                hint_literals = []
                for idx, name in enumerate(data.node_names):
                    if name in miter.inputs:
                        prob = pol_scores[idx].item()
                        var_id = miter.var_map.get(name)
                        
                        if var_id:
                            # Positive if prob > 0.5, negative otherwise
                            signed_lit = var_id if prob > 0.5 else -var_id
                            hint_literals.append(signed_lit)
                
                # 3. Solve with phase hints
                clauses_gnn = miter.build_miter(target_gate, fault_type, 1)
                
                with Minisat22(bootstrap_with=clauses_gnn) as solver:
                    if hint_literals:
                        solver.set_phases(hint_literals)
                    
                    gnn_result = solver.solve()
                    gnn_conflicts = solver.accum_stats()['conflicts']
                
                gnn_time = time.time() - t_gnn_start
                
                # Calculate speedup
                speedup = std_conflicts / max(gnn_conflicts, 1)
                status = "SAT" if gnn_result else "UNSAT"
                
                print(f"  Fault {i+1}/{num_tests} ({target_gate}): "
                      f"{std_conflicts} → {gnn_conflicts} conflicts ({speedup:.2f}x speedup)")
                
                # Record results
                results.append({
                    "Circuit": filename,
                    "Fault": target_gate,
                    "Status": status,
                    "Std_Conflicts": std_conflicts,
                    "GNN_Conflicts": gnn_conflicts,
                    "Speedup": speedup,
                    "Std_Time_s": std_time,
                    "GNN_Time_s": gnn_time
                })
        
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save results
    if results:
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)
        
        with open(RESULTS_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        # Print summary statistics
        total_tests = len(results)
        avg_speedup = sum(r['Speedup'] for r in results) / total_tests
        max_speedup = max(r['Speedup'] for r in results)
        min_speedup = min(r['Speedup'] for r in results)
        
        speedups = [r['Speedup'] for r in results]
        median_speedup = sorted(speedups)[len(speedups) // 2]
        
        # Count improvements
        improved = sum(1 for r in results if r['Speedup'] > 1.0)
        improved_pct = (improved / total_tests) * 100
        
        print(f"Total tests: {total_tests}")
        print(f"Tests improved: {improved} ({improved_pct:.1f}%)")
        print(f"\nSpeedup Statistics:")
        print(f"  Average:  {avg_speedup:.2f}x")
        print(f"  Median:   {median_speedup:.2f}x")
        print(f"  Min:      {min_speedup:.2f}x")
        print(f"  Max:      {max_speedup:.2f}x")
        print(f"\nResults saved to: {RESULTS_PATH}")
        print("=" * 80)
    else:
        print("\nNo results generated!")

if __name__ == "__main__":
    run_benchmark()