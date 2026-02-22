"""
Benchmark Script with FLEXIBLE SAMPLING OPTIONS

Supports:
1. All faults (comprehensive but slow)
2. Random sampling (quick statistical estimate)
3. Stratified sampling (representative across circuit structure)
4. First N faults (deterministic, repeatable)
"""

import os
import sys
import time
import csv
import random
import numpy as np
import torch
from pathlib import Path
from pysat.solvers import Minisat22, Glucose3

from WireFaultMiter import WireFaultMiter
from neuro_utils import VectorizedGraphExtractor
from train_model_2 import CircuitGNN_Polarity

# =============================================================================
# CONFIGS
# =============================================================================
BENCHMARK_DIR = "../../hdl-benchmarks/iscas85/bench/"
MODEL_PATH = "best_model.pt"
RESULTS_PATH = "results_gnn_polarity.csv"
CONFIDENCE_HIGH = 0.95
CONFIDENCE_LOW = 0.05

# SAMPLING OPTIONS
SAMPLING_MODE = "random"  # Options: "all", "random", "stratified", "first_n"
SAMPLE_SIZE = 50          # Number of faults to test per circuit (if not "all")
RANDOM_SEED = 42

# =============================================================================
# SAMPLING STRATEGIES
# =============================================================================

def sample_faults_all(all_gates):
    """Sample ALL faults (both SA0 and SA1 for each gate)"""
    faults = []
    for gate in all_gates:
        faults.append((gate, 0))  # SA0
        faults.append((gate, 1))  # SA1
    return faults


def sample_faults_random(all_gates, sample_size, seed=42):
    """Random sampling of faults"""
    random.seed(seed)
    
    # Generate all possible faults
    all_faults = []
    for gate in all_gates:
        all_faults.append((gate, 0))  # SA0
        all_faults.append((gate, 1))  # SA1
    
    # Sample randomly
    if len(all_faults) <= sample_size:
        return all_faults
    
    return random.sample(all_faults, sample_size)


def sample_faults_first_n(all_gates, sample_size):
    """Take first N faults (deterministic, repeatable)"""
    faults = []
    count = 0
    
    for gate in all_gates:
        if count >= sample_size:
            break
        faults.append((gate, 0))  # SA0
        count += 1
        
        if count >= sample_size:
            break
        faults.append((gate, 1))  # SA1
        count += 1
    
    return faults


def sample_faults_stratified(all_gates, sample_size, seed=42):
    """
    Stratified sampling: sample evenly across the circuit structure.
    
    Strategy:
    1. Divide gates into 3 groups: early (near inputs), middle, late (near outputs)
    2. Sample proportionally from each group
    3. Ensures coverage across different circuit depths
    """
    random.seed(seed)
    
    # For simplicity, divide gates into thirds
    n = len(all_gates)
    
    early_gates = all_gates[:n//3]
    middle_gates = all_gates[n//3:2*n//3]
    late_gates = all_gates[2*n//3:]
    
    # Sample from each group proportionally
    samples_per_group = sample_size // 3
    
    faults = []
    
    for group in [early_gates, middle_gates, late_gates]:
        group_faults = []
        for gate in group:
            group_faults.append((gate, 0))  # SA0
            group_faults.append((gate, 1))  # SA1
        
        # Sample from this group
        if len(group_faults) <= samples_per_group:
            faults.extend(group_faults)
        else:
            faults.extend(random.sample(group_faults, samples_per_group))
    
    return faults


def get_fault_sample(all_gates, mode, sample_size, seed=42):
    """
    Get fault sample based on mode.
    
    Args:
        all_gates: List of gate names
        mode: Sampling mode ("all", "random", "stratified", "first_n")
        sample_size: Number of faults to sample (ignored for "all")
        seed: Random seed for reproducibility
    
    Returns:
        List of (gate_name, fault_type) tuples
    """
    if mode == "all":
        return sample_faults_all(all_gates)
    elif mode == "random":
        return sample_faults_random(all_gates, sample_size, seed)
    elif mode == "stratified":
        return sample_faults_stratified(all_gates, sample_size, seed)
    elif mode == "first_n":
        return sample_faults_first_n(all_gates, sample_size)
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")


# =============================================================================
# BENCHMARKING FUNCTIONS
# =============================================================================

def benchmark_single_fault(miter, extractor, model, device, target_gate, fault_type):
    """Benchmark a single fault"""
    reachable = miter.get_reachable_outputs(target_gate)
    if not reachable: return None
    target_output = reachable[0]

    # Pass the target output to the miter
    clauses = miter.build_miter(target_gate, fault_type, force_diff=1, target_output=target_output)
    if not clauses: return None
    
    # Baseline
    t_std_start = time.time()
    with Glucose3(bootstrap_with=clauses) as solver:
        solver.conf_budget(100000)
        std_result = solver.solve()
        std_conflicts = solver.accum_stats()['conflicts']
    std_time = time.time() - t_std_start
    
    if not std_result:
        return None
    
    # GNN-guided
    t_gnn_start = time.time()
    complete_cone = miter.get_complete_atpg_cone(target_gate, target_output)

    data = extractor.get_data_for_fault(target_gate, fault_type=fault_type).to(device)
    
    with torch.no_grad():
        pol_scores = model(data)
    
    # === NEW: TOP-K HINT SELECTION ===
    # Instead of static thresholds, we collect all probabilities and sort them by confidence
    predictions = []
    cone_inputs = miter.get_cone_inputs(complete_cone)
    for idx, name in enumerate(data.node_names):
        if name in cone_inputs:
            prob = pol_scores[idx].item()
            var_id = miter.var_map.get(name)
            
            if var_id:
                # Confidence is how far the probability is from 0.5
                confidence = abs(prob - 0.5)
                is_high = prob > 0.5
                predictions.append((confidence, var_id, is_high))
    
    # Sort by highest confidence first
    predictions.sort(reverse=True, key=lambda x: x[0])
    
    # Take ONLY the top 5 most confident hints
    MAX_HINTS = 5
    top_predictions = predictions[:MAX_HINTS]
    
    hint_literals = []
    num_hints_high = 0
    num_hints_low = 0
    
    for confidence, var_id, is_high in top_predictions:
        # Only take it if it's at least somewhat confident (e.g. >0.75 or <0.25)
        if confidence > 0.25: 
            if is_high:
                hint_literals.append(var_id)
                num_hints_high += 1
            else:
                hint_literals.append(-var_id)
                num_hints_low += 1
                
    num_uncertain = len(miter.inputs) - (num_hints_high + num_hints_low)
    # ==================================
    
    with Minisat22(bootstrap_with=clauses) as solver:
        solver.conf_budget(100000)
        
        if hint_literals:
            solver.set_phases(hint_literals)
        
        gnn_result = solver.solve()
        gnn_conflicts = solver.accum_stats()['conflicts']
    
    gnn_time = time.time() - t_gnn_start
    
    speedup = std_conflicts / max(gnn_conflicts, 1)
    
    return {
        "fault_gate": target_gate,
        "fault_type": "SA1" if fault_type == 1 else "SA0",
        "std_conflicts": std_conflicts,
        "gnn_conflicts": gnn_conflicts,
        "speedup": speedup,
        "std_time": std_time,
        "gnn_time": gnn_time,
        "num_hints_high": num_hints_high,
        "num_hints_low": num_hints_low,
        "num_uncertain": num_uncertain,
        "total_inputs": len(miter.inputs)
    }


def benchmark_circuit(bench_file, model, device, sampling_mode, sample_size, seed):
    """Benchmark sampled faults in a circuit"""
    circuit_name = os.path.basename(bench_file).replace('.bench', '')
    print(f"\nBenchmarking {circuit_name}...")
    
    try:
        miter = WireFaultMiter(bench_file)
        num_gates = len(miter.gates)
        
        if num_gates == 0:
            print(f"  Skipping (no gates)")
            return []
        
        if num_gates > 10000:
            print(f"  Skipping (too large: {num_gates} gates)")
            return []
        
        print(f"  Gates: {num_gates}, Inputs: {len(miter.inputs)}")
        
        # Get gate names
        all_gates = [g[0] for g in miter.gates]
        
        # Sample faults based on mode
        fault_sample = get_fault_sample(all_gates, sampling_mode, sample_size, seed)
        
        print(f"  Sampling mode: {sampling_mode}")
        print(f"  Testing {len(fault_sample)} faults (out of {2*num_gates} possible)")
        
        # Create extractor
        extractor = VectorizedGraphExtractor(bench_file, var_map=miter.var_map, device=device.type)
        
        # Test sampled faults
        results = []
        
        for i, (target_gate, fault_type) in enumerate(fault_sample):
            result = benchmark_single_fault(
                miter, extractor, model, device, target_gate, fault_type
            )
            
            if result is not None:
                result['circuit'] = circuit_name
                results.append(result)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(fault_sample)} sampled faults, "
                      f"found {len(results)} testable")
        
        print(f"  Complete: {len(results)} testable faults")
        return results
    
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_benchmark():
    """Main benchmarking function"""
    print("=" * 80)
    print("GNN-GUIDED SAT SOLVING BENCHMARK (WITH SAMPLING)")
    print("=" * 80)
    print(f"Sampling mode: {SAMPLING_MODE}")
    if SAMPLING_MODE != "all":
        print(f"Sample size: {SAMPLE_SIZE} faults per circuit")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    model = CircuitGNN_Polarity(
        num_node_features=17,
        num_layers=checkpoint['args'].num_layers,
        hidden_dim=checkpoint['args'].hidden_dim,
        dropout=checkpoint['args'].dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Get benchmark files
    bench_path = Path(BENCHMARK_DIR)
    files = sorted(bench_path.glob('*.bench'))
    
    if not files:
        print(f"ERROR: No benchmark files found in {BENCHMARK_DIR}")
        return
    
    print(f"Found {len(files)} benchmark circuits")
    print("=" * 80)
    
    all_results = []
    
    for bench_file in files:
        circuit_results = benchmark_circuit(
            str(bench_file), model, device, 
            SAMPLING_MODE, SAMPLE_SIZE, RANDOM_SEED
        )
        all_results.extend(circuit_results)
    
    # Save and analyze results
    if all_results:
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)
        
        with open(RESULTS_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        
        # Statistics
        total_tests = len(all_results)
        speedups = [r['speedup'] for r in all_results]
        
        improved = sum(1 for s in speedups if s > 1.0)
        improved_pct = (improved / total_tests) * 100
        
        print(f"Total testable faults: {total_tests}")
        print(f"Faults with speedup: {improved} ({improved_pct:.1f}%)")
        print(f"\nSpeedup Statistics:")
        print(f"  Average:  {np.mean(speedups):.2f}x")
        print(f"  Median:   {np.median(speedups):.2f}x")
        print(f"  Min:      {np.min(speedups):.2f}x")
        print(f"  Max:      {np.max(speedups):.2f}x")
        
        print(f"\nPercentiles:")
        for p in [10, 25, 50, 75, 90]:
            print(f"  {p}th: {np.percentile(speedups, p):.2f}x")
        
        # Hint coverage
        avg_hints = np.mean([r['num_hints_high'] + r['num_hints_low'] for r in all_results])
        avg_total = np.mean([r['total_inputs'] for r in all_results])
        print(f"\nHint coverage: {avg_hints/avg_total*100:.1f}%")
        
        print(f"\nResults saved to: {RESULTS_PATH}")
        print("=" * 80)
    else:
        print("\nNo results generated!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark GNN-guided ATPG with sampling')
    parser.add_argument('--mode', type=str, default="random",
                       choices=["all", "random", "stratified", "first_n"],
                       help='Sampling mode')
    parser.add_argument('--sample_size', type=int, default=50,
                       help='Number of faults to sample per circuit (ignored for "all")')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--model', type=str, default="best_model.pt",
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default="results_gnn_polarity.csv",
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Update globals
    SAMPLING_MODE = args.mode
    SAMPLE_SIZE = args.sample_size
    RANDOM_SEED = args.seed
    MODEL_PATH = args.model
    RESULTS_PATH = args.output
    
    run_benchmark()