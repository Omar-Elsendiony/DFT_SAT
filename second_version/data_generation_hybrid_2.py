"""
ROBUST data generation with PyTorch pickling fix

Key fix: Workers return plain Python dicts instead of PyTorch tensors
Main process converts to PyTorch tensors after collecting results
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

CONFLICT_BUDGET = 10000
CRITICAL_INPUT_TEST_BUDGET = 20
PER_FAULT_TIMEOUT = 300  # 5 minutes per fault


def identify_critical_inputs_adaptive(clauses, assignment, cone_inputs, var_map):
    """ADAPTIVE critical input identification"""
    critical_inputs = {}
    
    if not cone_inputs:
        return critical_inputs
    
    test_inputs = []
    for inp in cone_inputs:
        if inp not in var_map:
            continue
        var_id = var_map[inp]
        correct_polarity = var_id in assignment
        test_literal = -var_id if correct_polarity else var_id
        test_inputs.append((inp, var_id, correct_polarity, test_literal))
    
    if not test_inputs:
        return critical_inputs
    
    random.shuffle(test_inputs)
    
    try:
        with Glucose3(bootstrap_with=clauses) as probe:
            tested_count = 0
            tests_since_last_critical = 0
            
            for inp, var_id, correct_polarity, test_literal in test_inputs:
                probe.conf_budget(CRITICAL_INPUT_TEST_BUDGET)
                
                result = probe.solve(assumptions=[test_literal])
                tested_count += 1
                tests_since_last_critical += 1
                
                if not result:
                    critical_inputs[inp] = 1.0 if correct_polarity else 0.0
                    tests_since_last_critical = 0
                
                # Early termination
                if len(critical_inputs) >= 3 and tests_since_last_critical >= 3:
                    break
                
                if tested_count >= 8 and len(critical_inputs) <= 2:
                    break
                
                if len(critical_inputs) >= 5:
                    break
    except Exception as e:
        pass
    
    return critical_inputs


def process_single_fault(args):
    """
    Process single fault - Returns PLAIN PYTHON DICT (not PyTorch tensors!)
    
    This avoids PyTorch tensor pickling issues in multiprocessing.
    """
    bench_file, fault_name, fault_type = args
    
    try:
        # Parse circuit
        if bench_file.endswith('.bench'):
            parser = BenchParser(bench_file)
        else:
            parser = VerilogParser(bench_file)
        
        # Create fault miter
        miter = WireFaultMiter(bench_file)
        clauses = miter.build_miter(fault_name, fault_type, force_diff=1)
        
        if not clauses:
            return None
        
        # Get complete ATPG cone
        reachable = miter.get_reachable_outputs(fault_name)
        if not reachable:
            return None
        
        target_output = reachable[0]
        complete_cone = miter.get_complete_atpg_cone(fault_name, target_output)
        
        if not complete_cone:
            return None
        
        # Solve to find if fault is testable
        with Glucose3(bootstrap_with=clauses) as solver:
            solver.conf_budget(CONFLICT_BUDGET)
            if not solver.solve():
                return None
            
            assignment = set(solver.get_model())
        
        # Get inputs in the cone
        cone_inputs = miter.get_cone_inputs(complete_cone)
        if not cone_inputs:
            return None
        
        # Identify critical inputs
        critical_inputs = identify_critical_inputs_adaptive(
            clauses, assignment, cone_inputs, miter.var_map
        )
        
        if len(critical_inputs) < 1:
            return None
        
        # Get graph data
        extractor = VectorizedGraphExtractor(bench_file, var_map=miter.var_map, device='cpu')
        data = extractor.get_data_for_fault(fault_name, fault_type=fault_type)
        
        # CRITICAL FIX: Convert to plain Python dict instead of returning PyTorch Data
        result = {
            'node_names': list(data.node_names),  # List of strings
            'x': data.x.cpu().numpy(),  # Convert to numpy array
            'edge_index': data.edge_index.cpu().numpy(),  # Convert to numpy
            'critical_inputs': critical_inputs,  # Already a dict
            'fault_name': fault_name,
            'fault_type': fault_type,
            'num_critical_inputs': len(critical_inputs)
        }
        
        return result
        
    except Exception as e:
        # Silently skip problematic faults
        return None


def dict_to_data(result_dict):
    """
    Convert plain Python dict back to PyTorch Data object.
    
    This is done in the main process, not in workers, so no pickling issues.
    """
    # Convert numpy arrays back to tensors
    x = torch.from_numpy(result_dict['x']).float()
    edge_index = torch.from_numpy(result_dict['edge_index']).long()
    
    # Create labels
    node_names = result_dict['node_names']
    critical_inputs = result_dict['critical_inputs']
    
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
    data.num_critical_inputs = result_dict['num_critical_inputs']
    
    return data


def generate_dataset_parallel(bench_file, output_dir, num_workers=4, save_interval=100):
    """Generate dataset with PyTorch pickling fix and incremental saving"""
    
    # Parse circuit once
    if bench_file.endswith('.bench'):
        parser = BenchParser(bench_file)
    else:
        parser = VerilogParser(bench_file)
    
    all_gates = list(parser.gate_dict.keys())
    
    # Generate fault list
    fault_list = []
    for gate in all_gates:
        fault_list.append((bench_file, gate, 0))
        fault_list.append((bench_file, gate, 1))
    
    print(f"Processing {len(fault_list)} faults using {num_workers} workers (ROBUST ADAPTIVE mode)...")
    
    # Setup save paths
    os.makedirs(output_dir, exist_ok=True)
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
    
    # Use smaller chunksize to avoid deadlock
    with Pool(num_workers) as pool:
        async_result = pool.imap_unordered(process_single_fault, fault_list, chunksize=1)
        
        for i in range(len(fault_list)):
            try:
                # Get result with timeout
                result_dict = async_result.next(timeout=PER_FAULT_TIMEOUT)
                processed_count += 1
                
                # Convert dict to Data object in main process (no pickling!)
                if result_dict is not None:
                    data = dict_to_data(result_dict)
                    dataset.append(data)
                
                # INCREMENTAL SAVE: Save every 100 new samples
                if len(dataset) - last_save_count >= save_interval:
                    try:
                        # Save to temp file first
                        with open(temp_save_path, 'wb') as f:
                            pickle.dump(dataset, f)
                        # Then rename to actual file (atomic operation)
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
                    
                    print(f"Processed {processed_count}/{len(fault_list)} faults, "
                          f"collected {len(dataset)} samples "
                          f"(~{rate:.1f} faults/s, ETA: {eta_seconds/60:.1f} min)")
                    
                    last_progress_time = current_time
                    last_count = processed_count
                
            except MPTimeoutError:
                processed_count += 1
                print(f"  Warning: Fault {processed_count} timed out after {PER_FAULT_TIMEOUT}s, skipping...")
                continue
            except StopIteration:
                break
            except Exception as e:
                processed_count += 1
                print(f"  Warning: Error processing fault {processed_count}: {e}")
                continue
    
    print(f"\nDataset generation complete!")
    print(f"Total faults processed: {processed_count}/{len(fault_list)}")
    print(f"Total samples collected: {len(dataset)}")
    
    # Final save
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved final dataset to {save_path}")
        
        # Clean up temp file if it exists
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
    
    return dataset


def generate_dataset_for_folder(bench_folder, output_dir, num_workers=4):
    """Generate dataset for all circuits in folder"""
    from pathlib import Path
    
    bench_folder = Path(bench_folder)
    bench_files = list(bench_folder.rglob('*.bench')) + list(bench_folder.rglob('*.v'))
    
    if not bench_files:
        print(f"No .bench or .v files found in {bench_folder}")
        return 0
    
    print(f"Found {len(bench_files)} circuits in {bench_folder}")
    bench_files = sorted(bench_files)
    
    total_samples = 0
    start_time = time.time()
    
    for i, bench_file in enumerate(bench_files):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(bench_files)}] Processing {bench_file.name}...")
        print(f"{'='*70}")
        
        circuit_start = time.time()
        
        try:
            dataset = generate_dataset_parallel(str(bench_file), output_dir, num_workers)
            circuit_time = time.time() - circuit_start
            
            total_samples += len(dataset)
            print(f"Circuit completed in {circuit_time:.1f}s ({len(dataset)} samples)")
        except Exception as e:
            print(f"Error processing {bench_file.name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Continuing to next circuit...")
            continue
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"ALL CIRCUITS COMPLETE")
    print(f"{'='*70}")
    print(f"Total circuits processed: {len(bench_files)}")
    print(f"Total samples generated: {total_samples}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average per circuit: {total_time/len(bench_files):.1f}s")
    
    return total_samples


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Generate training data (FIXED multiprocessing)')
    parser.add_argument('--bench', type=str, required=True,
                       help='Path to .bench/.v file or folder containing them')
    parser.add_argument('--output', type=str, default='./training_data_critical',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout per fault in seconds')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save checkpoint every N samples')
    
    args = parser.parse_args()
    
    # Update global timeout
    PER_FAULT_TIMEOUT = args.timeout
    
    bench_path = Path(args.bench)
    
    if bench_path.is_dir():
        generate_dataset_for_folder(args.bench, args.output, args.workers)
    elif bench_path.is_file():
        generate_dataset_parallel(args.bench, args.output, args.workers, args.save_interval)
    else:
        print(f"Error: {args.bench} is not a valid file or directory")
        exit(1)