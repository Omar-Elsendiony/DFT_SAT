import os
import csv
import torch
import random
import time
import numpy as np
from pysat.solvers import Glucose3
from pysat.formula import CNF

# --- LOCAL IMPORTS ---
from WireFaultMiter import WireFaultMiter
from neuro_utils import FastGraphExtractor, CircuitGNN_Advanced

# --- CONFIG ---
BENCH_DIR = "../hdl-benchmarks/iscas89/bench/"  # Folder with your .bench files
MODEL_PATH = "gnn_model_oracle.pth" # The trained Oracle model
CSV_OUTPUT = "oracle_results.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Threshold: Only trigger GNN for faults that take at least X conflicts.
# Set to 0 to test EVERYTHING. Set to 500 to test only "Hard" faults.
HARDNESS_THRESHOLD = 0 

# Strategy: How many GNN suggestions to inject?
# Injecting too many might include a wrong guess. Injecting too few won't help enough.
TOP_K_ASSUMPTIONS = 5 

def get_target_files():
    if not os.path.exists(BENCH_DIR): return []
    return [f for f in os.listdir(BENCH_DIR) if f.endswith(".bench")]

def run_benchmark():
    print(f"--- BENCHMARKING NEURAL ORACLE ---")
    print(f"--- Model: {MODEL_PATH} on {device} ---")
    
    # Initialize Model (Must match training: 14 features)
    model = CircuitGNN_Advanced(num_node_features=14, num_layers=8, dropout=0.0)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run train_oracle.py first.")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    results = []

    files = get_target_files()
    if not files:
        print("No bench files found.")
        return

    # Process each circuit
    for filename in files:
        filepath = os.path.join(BENCH_DIR, filename)
        print(f"\nProcessing {filename}...")
        
        try:
            miter_base = WireFaultMiter(filepath)
            if not miter_base.gates: continue
            
            fast_extractor = FastGraphExtractor(filepath, miter_base.var_map)
            input_names = set(miter_base.inputs)

            # Test 20 random faults per circuit
            for i in range(20):
                target_gate = random.choice(miter_base.gates)[0]
                
                # --- 1. BASELINE (Standard SAT) ---
                clauses = miter_base.build_miter(target_gate, None, 1)
                cnf = CNF(); cnf.extend(clauses)
                
                t_std_start = time.time()
                std_conflicts = 0
                with Glucose3(bootstrap_with=cnf) as solver:
                    solver.solve()
                    std_conflicts = solver.accum_stats()['conflicts']
                std_time = time.time() - t_std_start
                
                # Skip trivial faults if configured
                if std_conflicts < HARDNESS_THRESHOLD:
                    continue

                # --- 2. NEURAL ORACLE (Guided SAT) ---
                t_gnn_start = time.time()
                
                # A. Run Inference
                data = fast_extractor.get_data_for_fault(target_gate)
                if data is None: continue
                data = data.to(device)
                
                with torch.no_grad():
                    logits = model(data)
                    probs = torch.sigmoid(logits) # Convert logits to 0.0-1.0
                
                # B. Extract & Rank Hints
                hints = []
                for idx, name in enumerate(data.node_names):
                    if name in input_names:
                        prob = probs[idx].item()
                        confidence = abs(prob - 0.5) # How sure is the GNN?
                        
                        # Direction: If > 0.5, guess 1. Else guess 0.
                        val = 1 if prob > 0.5 else -1
                        
                        var_id = miter_base.var_map.get(name) or miter_base.faulty_map.get(name)
                        if var_id:
                            hints.append((var_id, val, confidence))
                
                # Sort by Confidence (Highest first)
                hints.sort(key=lambda x: x[2], reverse=True)
                
                # Take top K assumptions
                top_guesses = [h[0] * h[1] for h in hints[:TOP_K_ASSUMPTIONS]]
                
                # C. Solve with Assumptions
                gnn_conflicts = 0
                solved_by_gnn = False
                
                with Glucose3(bootstrap_with=cnf) as solver:
                    # Try to solve using the GNN's "Golden Key"
                    if solver.solve(assumptions=top_guesses):
                        gnn_conflicts = solver.accum_stats()['conflicts']
                        solved_by_gnn = True
                    else:
                        # If GNN was wrong (UNSAT under assumptions), fallback to unguided
                        # Note: This adds penalty time!
                        solver.solve()
                        gnn_conflicts = solver.accum_stats()['conflicts'] + 1000 # Penalty for wrong guess

                gnn_total_time = time.time() - t_gnn_start

                # --- 3. REPORTING ---
                # Avoid division by zero
                safe_gnn = max(gnn_conflicts, 1)
                safe_time = max(gnn_total_time, 0.0001)
                
                conflict_red = std_conflicts / safe_gnn
                time_speedup = std_time / safe_time
                
                tag = "[SUCCESS]" if solved_by_gnn else "[MISGUIDED]"
                if conflict_red > 1.0: tag = "[WIN]"
                
                print(f"   {tag} Fault: {target_gate} | Conf: {hints[0][2]:.4f}")
                print(f"         Conflicts: {std_conflicts} -> {gnn_conflicts} ({conflict_red:.2f}x reduction)")
                print(f"         Time:      {std_time:.4f}s -> {gnn_total_time:.4f}s")

                results.append({
                    "Circuit": filename,
                    "Fault": target_gate,
                    "Std_Conflicts": std_conflicts,
                    "GNN_Conflicts": gnn_conflicts,
                    "Speedup": f"{time_speedup:.2f}",
                    "Status": "Solved" if solved_by_gnn else "Fallback"
                })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save to CSV
    if results:
        with open(CSV_OUTPUT, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nBenchmark Complete. Results saved to {CSV_OUTPUT}")

if __name__ == "__main__":
    run_benchmark()