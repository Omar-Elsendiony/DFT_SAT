"""
FIXED Benchmark Script for GNN-Guided Circuit ATPG

This version:
1. Uses the correct model architecture (CircuitGNN_Polarity)
2. Properly unpacks model output (single value, not tuple)
3. Uses confidence filtering (0.6/0.4 thresholds)
4. Tests ALL faults (not just 10 random ones)
5. Provides comprehensive statistics
"""

import os
import sys
import time
import csv
import numpy as np
import torch
from pathlib import Path
from pysat.solvers import Minisat22, Glucose3

# Your imports
from WireFaultMiter import WireFaultMiter
from neuro_utils import VectorizedGraphExtractor
from train_model import CircuitGNN_Polarity  # FIXED: Correct model

# =============================================================================
# CONFIGS
# =============================================================================
BENCHMARK_DIR = "../../hdl-benchmarks/iscas85/bench/"
MODEL_PATH = "models/best_model.pt"
RESULTS_PATH = "results_gnn_polarity_fixed.csv"
CONFIDENCE_HIGH = 0.6  # Only set TRUE hint if prob > this
CONFIDENCE_LOW = 0.4   # Only set FALSE hint if prob < this

# =============================================================================
# BENCHMARKING FUNCTIONS
# =============================================================================

def benchmark_single_fault(miter, extractor, model, device, target_gate, fault_type):
    """
    Benchmark a single fault with baseline and GNN-guided solving.
    
    Returns dict with results or None if fault is untestable.
    """
    # Build miter for this fault
    clauses = miter.build_miter(target_gate, fault_type, force_diff=1)
    
    if not clauses:
        return None
    
    # === BASELINE: Standard SAT solving ===
    t_std_start = time.time()
    with Glucose3(bootstrap_with=clauses) as solver:
        solver.conf_budget(100000)
        std_result = solver.solve()
        std_conflicts = solver.accum_stats()['conflicts']
    std_time = time.time() - t_std_start
    
    if not std_result:
        # Untestable fault
        return None
    
    # === GNN-GUIDED: With polarity hints ===
    t_gnn_start = time.time()
    
    # 1. Get GNN predictions
    data = extractor.get_data_for_fault(target_gate, fault_type=fault_type).to(device)
    
    with torch.no_grad():
        pol_scores = model(data)  # FIXED: Single output [N, 1]
    
    # 2. Build phase hints with CONFIDENCE FILTERING
    hint_literals = []
    num_hints_high = 0
    num_hints_low = 0
    num_uncertain = 0
    
    for idx, name in enumerate(data.node_names):
        if name in miter.inputs:
            prob = pol_scores[idx].item()
            var_id = miter.var_map.get(name)
            
            if var_id:
                # FIXED: Use confidence thresholds
                if prob > CONFIDENCE_HIGH:
                    hint_literals.append(var_id)  # Prefer TRUE
                    num_hints_high += 1
                elif prob < CONFIDENCE_LOW:
                    hint_literals.append(-var_id)  # Prefer FALSE
                    num_hints_low += 1
                else:
                    # Uncertain - don't set hint
                    num_uncertain += 1
    
    # 3. Solve with phase hints
    with Minisat22(bootstrap_with=clauses) as solver:
        solver.conf_budget(100000)
        
        if hint_literals:
            solver.set_phases(hint_literals)
        
        gnn_result = solver.solve()
        gnn_conflicts = solver.accum_stats()['conflicts']
    
    gnn_time = time.time() - t_gnn_start
    
    # Calculate speedup
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


def benchmark_circuit(bench_file, model, device):
    """
    Benchmark all faults in a circuit.
    
    Returns list of result dicts.
    """
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
        
        # Create extractor
        extractor = VectorizedGraphExtractor(bench_file, var_map=miter.var_map, device=device.type)
        
        # Test ALL gates with both fault types
        all_gates = [g[0] for g in miter.gates]
        results = []
        
        for i, target_gate in enumerate(all_gates):
            for fault_type in [0, 1]:  # SA0 and SA1
                result = benchmark_single_fault(
                    miter, extractor, model, device, target_gate, fault_type
                )
                
                if result is not None:
                    result['circuit'] = circuit_name
                    results.append(result)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(all_gates)} gates, "
                      f"found {len(results)} testable faults")
        
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
    print("GNN-GUIDED SAT SOLVING BENCHMARK (FIXED VERSION)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please train the model first!")
        print(f"Expected location: {MODEL_PATH}")
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Get model architecture from checkpoint
    model = CircuitGNN_Polarity(
        num_node_features=17,
        num_layers=checkpoint['args'].num_layers,
        hidden_dim=checkpoint['args'].hidden_dim,
        dropout=checkpoint['args'].dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")
    print(f"  Architecture: {checkpoint['args'].num_layers} layers, {checkpoint['args'].hidden_dim} hidden dims")
    
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
        circuit_results = benchmark_circuit(str(bench_file), model, device)
        all_results.extend(circuit_results)
    
    # Save and analyze results
    if all_results:
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)
        
        # Save to CSV
        with open(RESULTS_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        
        # Compute statistics
        total_tests = len(all_results)
        speedups = [r['speedup'] for r in all_results]
        
        avg_speedup = np.mean(speedups)
        median_speedup = np.median(speedups)
        max_speedup = np.max(speedups)
        min_speedup = np.min(speedups)
        std_speedup = np.std(speedups)
        
        # Count improvements
        improved = sum(1 for s in speedups if s > 1.0)
        improved_pct = (improved / total_tests) * 100
        
        print(f"Total testable faults: {total_tests}")
        print(f"Faults with speedup (>1.0x): {improved} ({improved_pct:.1f}%)")
        print(f"\nSpeedup Statistics:")
        print(f"  Average:  {avg_speedup:.2f}x")
        print(f"  Median:   {median_speedup:.2f}x")
        print(f"  Std Dev:  {std_speedup:.2f}")
        print(f"  Min:      {min_speedup:.2f}x")
        print(f"  Max:      {max_speedup:.2f}x")
        
        # Percentiles
        print(f"\nPercentiles:")
        for p in [10, 25, 50, 75, 90]:
            val = np.percentile(speedups, p)
            print(f"  {p}th: {val:.2f}x")
        
        # Hint statistics
        avg_hints_high = np.mean([r['num_hints_high'] for r in all_results])
        avg_hints_low = np.mean([r['num_hints_low'] for r in all_results])
        avg_uncertain = np.mean([r['num_uncertain'] for r in all_results])
        avg_total = np.mean([r['total_inputs'] for r in all_results])
        
        print(f"\nAverage hints per fault:")
        print(f"  High confidence (TRUE): {avg_hints_high:.1f}")
        print(f"  Low confidence (FALSE): {avg_hints_low:.1f}")
        print(f"  Uncertain (no hint): {avg_uncertain:.1f}")
        print(f"  Total inputs: {avg_total:.1f}")
        hint_coverage = (avg_hints_high + avg_hints_low) / avg_total * 100
        print(f"  Hint coverage: {hint_coverage:.1f}%")
        
        # Per-circuit summary
        print(f"\n{'='*80}")
        print("PER-CIRCUIT SUMMARY")
        print(f"{'='*80}")
        print(f"{'Circuit':<15} {'Faults':<10} {'Speedup Rate':<15} {'Avg Speedup':<15}")
        print(f"{'-'*80}")
        
        circuits = {}
        for r in all_results:
            if r['circuit'] not in circuits:
                circuits[r['circuit']] = []
            circuits[r['circuit']].append(r['speedup'])
        
        for circuit in sorted(circuits.keys()):
            speedups_circuit = circuits[circuit]
            improved_circuit = sum(1 for s in speedups_circuit if s > 1.0)
            rate = improved_circuit / len(speedups_circuit) * 100
            avg_speedup_circuit = np.mean(speedups_circuit)
            
            print(f"{circuit:<15} {len(speedups_circuit):<10} "
                  f"{rate:>6.1f}% ({improved_circuit:>3}/{len(speedups_circuit):<3}) "
                  f"{avg_speedup_circuit:>6.2f}x")
        
        print(f"\nResults saved to: {RESULTS_PATH}")
        print("=" * 80)
    else:
        print("\nNo results generated!")

if __name__ == "__main__":
    run_benchmark()