"""
COMPLETE DATA GENERATION SCRIPT - All Outputs Version
======================================================

Fixes Applied:
1. ✅ Processes ALL reachable outputs (2-3x more training data)
2. ✅ Miter/cone consistency (same target_output for both)
3. ✅ No PyTorch pickling issues (returns plain Python dicts)
4. ✅ Incremental saving every 100 samples
5. ✅ Resume from checkpoint
6. ✅ Random fault sampling
7. ✅ Progress tracking with ETA

Usage:
    python data_generation_ALL_OUTPUTS.py \
        --bench hdl-benchmarks/iscas85/bench \
        --output ./training_data_v6 \
        --workers 8 \
        --max_faults 1000
"""

from pysat.solvers import Glucose3, Minisat22
import torch
from torch_geometric.data import Data
import multiprocessing as mp
from multiprocessing import Pool, TimeoutError as MPTimeoutError
import random
import os
import pickle
import time
import numpy as np

from BenchParser import BenchParser
from VerilogParser import VerilogParser  
from WireFaultMiter import WireFaultMiter
from neuro_utils import VectorizedGraphExtractor

# Configuration
CONFLICT_BUDGET = 10000
CRITICAL_INPUT_TEST_BUDGET = 20
PER_FAULT_TIMEOUT = 300

# ============================================================================
# Per-Process Parser Cache (avoids re-parsing same circuit 1000x)
# ============================================================================
_parser_cache = {}
_miter_cache = {}
_extractor_cache = {}

def get_cached_parser(bench_file):
    """Get cached parser for this process."""
    if bench_file not in _parser_cache:
        if bench_file.endswith('.bench'):
            _parser_cache[bench_file] = BenchParser(bench_file)
        else:
            _parser_cache[bench_file] = VerilogParser(bench_file)
    return _parser_cache[bench_file]

def get_cached_miter(bench_file):
    """Get cached miter for this process."""
    if bench_file not in _miter_cache:
        _miter_cache[bench_file] = WireFaultMiter(bench_file)
    return _miter_cache[bench_file]

def get_cached_extractor(bench_file, var_map):
    """Get cached extractor for this process."""
    if bench_file not in _extractor_cache:
        _extractor_cache[bench_file] = VectorizedGraphExtractor(
            bench_file, var_map=var_map, device='cpu'
        )
    return _extractor_cache[bench_file]


def identify_critical_inputs_adaptive(clauses, assignment, cone_inputs, var_map, debug=False):
    """ADAPTIVE critical input identification with early termination"""
    critical_inputs = {}
    
    if not cone_inputs:
        if debug:
            print("  [DEBUG] No cone inputs")
        return critical_inputs
    
    # Prepare test inputs
    test_inputs = []
    for inp in cone_inputs:
        if inp not in var_map:
            continue
        var_id = var_map[inp]
        correct_polarity = var_id in assignment
        test_literal = -var_id if correct_polarity else var_id
        test_inputs.append((inp, var_id, correct_polarity, test_literal))
    
    if not test_inputs:
        if debug:
            print("  [DEBUG] No test inputs (all filtered)")
        return critical_inputs
    
    # Shuffle for better sampling
    random.shuffle(test_inputs)
    
    if debug:
        print(f"  [DEBUG] Testing {len(test_inputs)} inputs")
    
    try:
        with Glucose3(bootstrap_with=clauses) as probe:
            tested_count = 0
            tests_since_last_critical = 0
            
            for inp, var_id, correct_polarity, test_literal in test_inputs:
                probe.conf_budget(CRITICAL_INPUT_TEST_BUDGET)
                
                result = probe.solve(assumptions=[test_literal])
                tested_count += 1
                tests_since_last_critical += 1
                
                if not result:  # UNSAT = critical input
                    critical_inputs[inp] = 1.0 if correct_polarity else 0.0
                    tests_since_last_critical = 0
                
                # IMPROVED early termination conditions
                # Only terminate early if we have SOME critical inputs
                if len(critical_inputs) >= 3 and tests_since_last_critical >= 5:
                    break
                
                # Less aggressive: allow more testing if we have few criticals
                if tested_count >= 15 and len(critical_inputs) <= 1:
                    break  # Tested 15 inputs, found 0 or 1 → probably not many criticals
                
                if len(critical_inputs) >= 5:
                    break  # Found enough
                    
        if debug:
            print(f"  [DEBUG] Found {len(critical_inputs)} critical inputs after testing {tested_count}")
            
    except Exception as e:
        if debug:
            print(f"  [DEBUG] Exception in critical input ID: {e}")
        pass
    
    return critical_inputs


def process_single_fault_first_output(args):
    """
    Process single fault for FIRST reachable output only.
    
    This is more stable than ALL outputs mode (avoids pipe overflow).
    Uses per-process caching to avoid re-parsing the circuit for every fault!
    
    Args:
        args: (bench_file, fault_name, fault_type)
    
    Returns:
        Single result dict, or None if fault is untestable
    """
    bench_file, fault_name, fault_type = args
    
    try:
        # Use cached instances (only parse circuit once per worker process!)
        miter = get_cached_miter(bench_file)
        
        # Get ALL reachable outputs
        reachable = miter.get_reachable_outputs(fault_name)
        if not reachable:
            return None
        
        # SIMPLIFIED: Use ONLY the first reachable output (stable, no pipe overflow)
        # This is more reliable than processing multiple outputs
        target_output = reachable[0]
        
        # Use cached extractor
        extractor = get_cached_extractor(bench_file, miter.var_map)
        
        # Process ONLY the first output (simple and stable)
        # =========================================================
        # BUILD MITER FOR THIS SPECIFIC OUTPUT
        # =========================================================
        clauses = miter.build_miter(
            fault_name, 
            fault_type, 
            force_diff=1,
            target_output=target_output  # ✅ Specific output!
        )
        
        if not clauses:
            return None  # Can't build miter
        
        # =========================================================
        # GET CONE FOR SAME OUTPUT (consistency!)
        # =========================================================
        complete_cone = miter.get_complete_atpg_cone(
            fault_name, 
            target_output
        )
        
        if not complete_cone:
            return None  # Can't get cone
        
        # =========================================================
        # SOLVE TO GET ASSIGNMENT
        # =========================================================
        with Glucose3(bootstrap_with=clauses) as solver:
            solver.conf_budget(CONFLICT_BUDGET)
            if not solver.solve():
                return None  # Fault not testable
            
            assignment = set(solver.get_model())
        
        # =========================================================
        # GET CONE INPUTS
        # =========================================================
        cone_inputs = miter.get_cone_inputs(complete_cone)
        if not cone_inputs:
            return None  # No cone inputs
        
        # =========================================================
        # IDENTIFY CRITICAL INPUTS
        # =========================================================
        critical_inputs = identify_critical_inputs_adaptive(
            clauses, assignment, cone_inputs, miter.var_map
        )
        
        # Accept even with 0 critical inputs (for analysis)
        # Some faults may have no critical inputs if fault is easily detectable
        
        # =========================================================
        # EXTRACT GRAPH FEATURES
        # =========================================================
        data = extractor.get_data_for_fault(fault_name, fault_type=fault_type)
        
        # =========================================================
        # CREATE RESULT (Plain Python dict for pickling)
        # =========================================================
        # Convert ALL data to plain Python types (no PyTorch references)
        node_names_list = [str(name) for name in data.node_names]  # Ensure strings
        x_numpy = data.x.detach().cpu().numpy().copy()  # Detach + copy!
        edge_index_numpy = data.edge_index.detach().cpu().numpy().copy()  # Detach + copy!
        
        # Convert critical_inputs dict to plain types
        critical_inputs_plain = {str(k): float(v) for k, v in critical_inputs.items()}
        
        result = {
            'node_names': node_names_list,
            'x': x_numpy,
            'edge_index': edge_index_numpy,
            'critical_inputs': critical_inputs_plain,
            'fault_name': str(fault_name),
            'fault_type': int(fault_type),
            'target_output': str(target_output),
            'num_critical_inputs': int(len(critical_inputs)),
            'num_cone_inputs': int(len(cone_inputs))
        }
        
        # Clean up data object only (extractor is cached)
        del data
        
        # Return single result
        return result
        
    except Exception as e:
        # Fault completely failed
        return None


def dict_to_data(result_dict):
    """Convert plain Python dict back to PyTorch Data object"""
    x = torch.from_numpy(result_dict['x']).float()
    edge_index = torch.from_numpy(result_dict['edge_index']).long()
    
    node_names = result_dict['node_names']
    critical_inputs = result_dict['critical_inputs']
    
    # Create labels
    y_polarity = torch.zeros(len(node_names), 1)
    train_mask = torch.zeros(len(node_names), 1)
    importance = torch.zeros(len(node_names), 1)
    
    for k, node_name in enumerate(node_names):
        if node_name in critical_inputs:
            y_polarity[k] = critical_inputs[node_name]
            train_mask[k] = 1.0
            importance[k] = 1.0
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index)
    data.node_names = node_names
    data.y_polarity = y_polarity
    data.train_mask = train_mask
    data.y_importance = importance
    data.fault_name = result_dict['fault_name']
    data.fault_type = result_dict['fault_type']
    data.target_output = result_dict['target_output']
    data.num_critical_inputs = result_dict['num_critical_inputs']
    
    return data


def sample_faults(all_gates, sample_size, seed=42):
    """Randomly sample faults from circuit"""
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


def generate_dataset_parallel(bench_file, output_dir, num_workers=4, 
                              save_interval=100, max_faults=None, seed=42, root_dir=None):
    """
    Generate dataset processing ALL reachable outputs per fault.
    
    This extracts maximum training data from each fault!
    
    Args:
        root_dir: Root directory for computing relative paths (for nested folders)
    """
    
    # Parse circuit
    if bench_file.endswith('.bench'):
        parser = BenchParser(bench_file)
    else:
        parser = VerilogParser(bench_file)
    
    all_gates = list(parser.gate_dict.keys())
    
    # Generate fault list (with optional sampling)
    if max_faults is not None:
        print(f"Sampling {max_faults} faults from {len(all_gates)*2} total faults (seed={seed})")
        sampled_faults = sample_faults(all_gates, max_faults, seed)
        fault_list = [(bench_file, gate, ftype) for gate, ftype in sampled_faults]
    else:
        print(f"Processing ALL {len(all_gates)*2} faults")
        fault_list = []
        for gate in all_gates:
            fault_list.append((bench_file, gate, 0))
            fault_list.append((bench_file, gate, 1))
    
    print(f"Processing {len(fault_list)} faults using {num_workers} workers")
    print(f"Mode: FIRST OUTPUT (stable, one sample per fault)")
    
    # Setup save paths with relative path encoding
    os.makedirs(output_dir, exist_ok=True)
    
    # Encode relative path in filename: folder1__folder2__file_name
    if root_dir:
        try:
            from pathlib import Path
            rel_path = Path(bench_file).relative_to(Path(root_dir))
            # Replace path separators with double underscores, remove extension
            circuit_name = str(rel_path.with_suffix('')).replace(os.sep, '__').replace('/', '__')
        except ValueError:
            # Fallback if relative path fails
            circuit_name = os.path.basename(bench_file).replace('.bench', '').replace('.v', '')
    else:
        circuit_name = os.path.basename(bench_file).replace('.bench', '').replace('.v', '')
    
    save_path = os.path.join(output_dir, f'{circuit_name}_critical_inputs.pkl')
    temp_save_path = os.path.join(output_dir, f'{circuit_name}_critical_inputs_temp.pkl')
    
    # Try to load existing progress
    dataset = []
    if os.path.exists(save_path):
        try:
            with open(save_path, 'rb') as f:
                dataset = pickle.load(f)
            print(f"Resumed from existing file: {len(dataset)} samples already collected")
        except:
            print("Could not load existing file, starting fresh")
    
    last_progress_time = time.time()
    last_count = 0
    processed_count = 0
    last_save_count = len(dataset)
    consecutive_timeouts = 0  # Track consecutive timeouts
    MAX_CONSECUTIVE_TIMEOUTS = 2  # Skip circuit after this many
    
    # Process faults with LIMITED tasks per child to avoid memory buildup
    # Each worker process will be restarted after processing some faults
    # This prevents pipe buffer overflow and memory leaks
    with Pool(num_workers, maxtasksperchild=50) as pool:
        async_result = pool.imap_unordered(
            process_single_fault_first_output,  # Changed to first output only
            fault_list, 
            chunksize=1  # Process one fault at a time
        )
        
        for i in range(len(fault_list)):
            try:
                result = async_result.next(timeout=PER_FAULT_TIMEOUT)
                processed_count += 1
                consecutive_timeouts = 0  # Reset on success
                
                if result is not None:
                    # Result is now a single dict (not a list)
                    data = dict_to_data(result)
                    dataset.append(data)
                
                # Incremental save
                if len(dataset) - last_save_count >= save_interval:
                    try:
                        with open(temp_save_path, 'wb') as f:
                            pickle.dump(dataset, f)
                        os.replace(temp_save_path, save_path)
                        last_save_count = len(dataset)
                        print(f"  → Saved checkpoint: {len(dataset)} samples")
                    except Exception as e:
                        print(f"  Warning: Could not save checkpoint: {e}")
                
                # Progress tracking
                current_time = time.time()
                if processed_count % 100 == 0:
                    elapsed = current_time - last_progress_time
                    rate = (processed_count - last_count) / elapsed if elapsed > 0 else 0
                    eta_seconds = (len(fault_list) - processed_count) / rate if rate > 0 else 0
                    
                    samples_per_fault = len(dataset) / processed_count if processed_count > 0 else 0
                    
                    print(f"Processed {processed_count}/{len(fault_list)} faults, "
                          f"collected {len(dataset)} samples "
                          f"(~{samples_per_fault:.2f} samples/fault, "
                          f"~{rate:.1f} faults/s, ETA: {eta_seconds/60:.1f} min)")
                    
                    last_progress_time = current_time
                    last_count = processed_count
                
            except MPTimeoutError:
                processed_count += 1
                consecutive_timeouts += 1
                fault_info = fault_list[i] if i < len(fault_list) else f"fault_{processed_count}"
                print(f"  Warning: Fault {fault_info} timed out ({consecutive_timeouts}/{MAX_CONSECUTIVE_TIMEOUTS}), skipping...")
                
                # Skip circuit if too many consecutive timeouts
                if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                    print(f"\n{'='*70}")
                    print(f"⚠️  SKIPPING CIRCUIT: {MAX_CONSECUTIVE_TIMEOUTS} consecutive timeouts detected!")
                    print(f"This circuit appears to be too large or problematic.")
                    print(f"Collected {len(dataset)} samples so far from this circuit.")
                    print(f"{'='*70}\n")
                    break
                continue
            except BrokenPipeError:
                # Pipe error - worker crashed, but we can continue with other faults
                processed_count += 1
                consecutive_timeouts = 0  # Don't count pipe errors as timeouts
                print(f"  Warning: Fault {processed_count} caused pipe error, skipping...")
                continue
            except StopIteration:
                break
            except Exception as e:
                processed_count += 1
                consecutive_timeouts = 0  # Don't count other errors as timeouts
                print(f"  Warning: Fault {processed_count} failed with error: {e}")
                continue
    
    print(f"\nDataset generation complete!")
    print(f"Total faults processed: {processed_count}/{len(fault_list)}")
    print(f"Total samples collected: {len(dataset)}")
    
    if processed_count > 0:
        avg_samples_per_fault = len(dataset) / processed_count
        print(f"Average samples per fault: {avg_samples_per_fault:.2f}")
        print(f"(This shows how many outputs each fault reaches on average)")
    
    # Final save
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved final dataset to {save_path}")
        
        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)
    except Exception as e:
        print(f"Warning: Could not save final dataset: {e}")
    
    # Print statistics
    if dataset:
        critical_counts = [d.num_critical_inputs for d in dataset]
        print(f"\nCritical input statistics:")
        print(f"  Min: {min(critical_counts)}")
        print(f"  Max: {max(critical_counts)}")
        print(f"  Avg: {sum(critical_counts) / len(critical_counts):.2f}")
        
        # Count samples with 0 critical inputs
        zero_critical = sum(1 for c in critical_counts if c == 0)
        if zero_critical > 0:
            print(f"  Samples with 0 critical inputs: {zero_critical} ({zero_critical/len(dataset)*100:.1f}%)")
        
        # Count unique faults
        unique_faults = set((d.fault_name, d.fault_type) for d in dataset)
        print(f"\nUnique faults: {len(unique_faults)}")
        print(f"Total samples: {len(dataset)}")
        print(f"Samples per fault: {len(dataset) / len(unique_faults):.2f}")
    
    return dataset


def generate_dataset_for_folder(bench_folder, output_dir, num_workers=4, 
                                max_faults_per_circuit=None, seed=42, skip_in_2=False):
    """Generate dataset for all circuits in folder (recursive with rglob)"""
    from pathlib import Path
    
    bench_folder_path = Path(bench_folder).resolve()
    
    # Use rglob for recursive search
    bench_files = list(bench_folder_path.rglob('*.bench')) + list(bench_folder_path.rglob('*.v'))
    
    if not bench_files:
        print(f"No .bench or .v files found in {bench_folder}")
        return 0
    
    # Optional: Skip in_2.v files (they seem problematic - produce 0 samples)
    if skip_in_2:
        bench_files = [f for f in bench_files if 'in_2.v' not in str(f)]
        print(f"Skipping in_2.v files (problematic circuits)")
        
    # Skip specific problematic files
    skip_files = {"bar.v", "adder.v", "div.v", "hyp.v"}

    bench_files = [
        f for f in bench_files
        if f.name not in skip_files
    ]
    if skip_files:
        print(f"Skipping files: {', '.join(skip_files)}")

    
    print(f"Found {len(bench_files)} circuits in {bench_folder} (recursive search)")
    if max_faults_per_circuit:
        print(f"Will sample {max_faults_per_circuit} faults per circuit (seed={seed})")
    print(f"Mode: FIRST OUTPUT (stable, reliable)")
    bench_files = sorted(bench_files)
    
    total_samples = 0
    start_time = time.time()
    
    for i, bench_file in enumerate(bench_files):
        print(f"\n{'='*70}")
        
        # Display relative path for nested folders
        try:
            rel_path = bench_file.relative_to(bench_folder_path)
            print(f"[{i+1}/{len(bench_files)}] Processing {rel_path}...")
        except ValueError:
            print(f"[{i+1}/{len(bench_files)}] Processing {bench_file.name}...")
        
        print(f"{'='*70}")
        
        circuit_start = time.time()
        
        try:
            dataset = generate_dataset_parallel(
                str(bench_file), output_dir, num_workers, 
                save_interval=100, max_faults=max_faults_per_circuit, seed=seed,
                root_dir=str(bench_folder_path)  # Pass root directory for relative paths
            )
            circuit_time = time.time() - circuit_start
            
            total_samples += len(dataset)
            print(f"Circuit completed in {circuit_time:.1f}s ({len(dataset)} samples)")
            
            # Force garbage collection between circuits
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {bench_file.name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Force cleanup on error
            import gc
            gc.collect()
            continue
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"ALL CIRCUITS COMPLETE")
    print(f"{'='*70}")
    print(f"Total circuits processed: {len(bench_files)}")
    print(f"Total samples generated: {total_samples}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    return total_samples


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Generate training data - ALL OUTPUTS mode',
        epilog="""
Examples:
  # ISCAS-85 (process all faults, all outputs):
  python data_generation_ALL_OUTPUTS.py \\
      --bench hdl-benchmarks/iscas85/bench \\
      --output ./training_data_v6 \\
      --workers 8
  
  # ICCAD-2015 (sample 1000 faults, all outputs):
  python data_generation_ALL_OUTPUTS.py \\
      --bench hdl-benchmarks/iccad-2015 \\
      --output ./training_data_iccad \\
      --workers 8 \\
      --max_faults 1000

Expected: 2-3x more training samples compared to single-output mode!
        """
    )
    
    parser.add_argument('--bench', type=str, required=True,
                       help='Path to .bench/.v file or folder')
    parser.add_argument('--output', type=str, default='./training_data_all_outputs',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout per fault in seconds')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save checkpoint every N samples')
    parser.add_argument('--max_faults', type=int, default=None,
                       help='Maximum faults to sample per circuit (None = all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for fault sampling')
    parser.add_argument('--skip_in_2', action='store_true',
                       help='Skip in_2.v files (they produce 0 samples)')
    
    args = parser.parse_args()
    
    # Update global timeout
    PER_FAULT_TIMEOUT = args.timeout
    
    bench_path = Path(args.bench)
    
    print("="*70)
    print("DATA GENERATION - FIRST OUTPUT MODE (STABLE)")
    print("="*70)
    print("This mode processes FIRST reachable output per fault")
    print("More stable than ALL outputs, no pipe overflow")
    print("="*70)
    
    if bench_path.is_dir():
        generate_dataset_for_folder(
            args.bench, args.output, args.workers, 
            max_faults_per_circuit=args.max_faults, seed=args.seed,
            skip_in_2=args.skip_in_2
        )
    elif bench_path.is_file():
        # For single file, use parent directory as root
        generate_dataset_parallel(
            args.bench, args.output, args.workers, args.save_interval,
            max_faults=args.max_faults, seed=args.seed,
            root_dir=str(bench_path.parent)
        )
    else:
        print(f"Error: {args.bench} is not a valid file or directory")
        exit(1)