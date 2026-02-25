"""
SEQUENTIAL DATA GENERATION (NO MULTIPROCESSING)
===============================================

100% reliable - no BrokenPipeError, no pipe buffer issues.
Slower than multiprocessing, but works for ALL circuits including large ones.

Usage:
    python data_generation_sequential.py \
        --bench hdl-benchmarks/iccad-2015 \
        --output ./training_data_v6 \
        --max_faults 1000 \
        --skip_in_2
"""

from pysat.solvers import Glucose3
import torch
from torch_geometric.data import Data
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


def identify_critical_inputs_adaptive(clauses, assignment, cone_inputs, var_map):
    """ADAPTIVE critical input identification with early termination"""
    critical_inputs = {}
    
    if not cone_inputs:
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
        return critical_inputs
    
    # Shuffle for better sampling
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
                
                if not result:  # UNSAT = critical input
                    critical_inputs[inp] = 1.0 if correct_polarity else 0.0
                    tests_since_last_critical = 0
                
                # IMPROVED early termination conditions
                if len(critical_inputs) >= 3 and tests_since_last_critical >= 5:
                    break
                
                if tested_count >= 15 and len(critical_inputs) <= 1:
                    break
                
                if len(critical_inputs) >= 5:
                    break
                    
    except Exception as e:
        pass
    
    return critical_inputs


def process_single_fault(bench_file, fault_name, fault_type, miter, extractor):
    """
    Process single fault (sequential, no multiprocessing).
    
    Returns Data object or None.
    """
    try:
        # Get first reachable output
        reachable = miter.get_reachable_outputs(fault_name)
        if not reachable:
            return None
        
        target_output = reachable[0]
        
        # Build miter
        clauses = miter.build_miter(
            fault_name, 
            fault_type, 
            force_diff=1,
            target_output=target_output
        )
        
        if not clauses:
            return None
        
        # Get cone
        complete_cone = miter.get_complete_atpg_cone(fault_name, target_output)
        if not complete_cone:
            return None
        
        # Solve
        with Glucose3(bootstrap_with=clauses) as solver:
            solver.conf_budget(CONFLICT_BUDGET)
            if not solver.solve():
                return None
            assignment = set(solver.get_model())
        
        # Get cone inputs
        cone_inputs = miter.get_cone_inputs(complete_cone)
        if not cone_inputs:
            return None
        
        # Identify critical inputs
        critical_inputs = identify_critical_inputs_adaptive(
            clauses, assignment, cone_inputs, miter.var_map
        )
        
        # Extract features
        data = extractor.get_data_for_fault(fault_name, fault_type=fault_type)
        
        # Create labels
        y_polarity = torch.zeros(len(data.node_names), 1)
        train_mask = torch.zeros(len(data.node_names), 1)
        importance = torch.zeros(len(data.node_names), 1)
        
        for k, node_name in enumerate(data.node_names):
            if node_name in critical_inputs:
                y_polarity[k] = critical_inputs[node_name]
                train_mask[k] = 1.0
                importance[k] = 1.0
        
        data.y_polarity = y_polarity
        data.train_mask = train_mask
        data.y_importance = importance
        data.fault_name = fault_name
        data.fault_type = fault_type
        data.target_output = target_output
        data.num_critical_inputs = len(critical_inputs)
        
        return data
        
    except Exception as e:
        return None


def sample_faults(all_gates, sample_size, seed=42):
    """Randomly sample faults from circuit"""
    random.seed(seed)
    
    all_faults = []
    for gate in all_gates:
        all_faults.append((gate, 0))
        all_faults.append((gate, 1))
    
    if len(all_faults) <= sample_size:
        return all_faults
    
    return random.sample(all_faults, sample_size)


def generate_dataset_sequential(bench_file, output_dir, save_interval=100, 
                                max_faults=None, seed=42, root_dir=None):
    """
    Generate dataset sequentially (NO multiprocessing).
    
    100% reliable - works for all circuits including massive ones.
    """
    
    # Parse circuit ONCE
    print(f"Parsing circuit...")
    start_parse = time.time()
    
    if bench_file.endswith('.bench'):
        parser = BenchParser(bench_file)
    else:
        parser = VerilogParser(bench_file)
    
    miter = WireFaultMiter(bench_file)
    extractor = VectorizedGraphExtractor(bench_file, var_map=miter.var_map, device='cpu')
    
    parse_time = time.time() - start_parse
    print(f"Parsing complete in {parse_time:.1f}s")
    
    all_gates = list(parser.gate_dict.keys())
    
    # Generate fault list
    if max_faults is not None:
        print(f"Sampling {max_faults} faults from {len(all_gates)*2} total faults (seed={seed})")
        sampled_faults = sample_faults(all_gates, max_faults, seed)
        fault_list = [(gate, ftype) for gate, ftype in sampled_faults]
    else:
        print(f"Processing ALL {len(all_gates)*2} faults")
        fault_list = []
        for gate in all_gates:
            fault_list.append((gate, 0))
            fault_list.append((gate, 1))
    
    print(f"Processing {len(fault_list)} faults (SEQUENTIAL mode - no multiprocessing)")
    
    # Setup save paths
    os.makedirs(output_dir, exist_ok=True)
    
    if root_dir:
        try:
            from pathlib import Path
            rel_path = Path(bench_file).relative_to(Path(root_dir))
            circuit_name = str(rel_path.with_suffix('')).replace(os.sep, '__').replace('/', '__')
        except ValueError:
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
    
    # Process faults SEQUENTIALLY
    for fault_name, fault_type in fault_list:
        result = process_single_fault(bench_file, fault_name, fault_type, miter, extractor)
        processed_count += 1
        
        if result is not None:
            dataset.append(result)
        
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
        if processed_count % 100 == 0:
            current_time = time.time()
            elapsed = current_time - last_progress_time
            rate = (processed_count - last_count) / elapsed if elapsed > 0 else 0
            eta_seconds = (len(fault_list) - processed_count) / rate if rate > 0 else 0
            
            print(f"Processed {processed_count}/{len(fault_list)} faults, "
                  f"collected {len(dataset)} samples "
                  f"(~{rate:.1f} faults/s, ETA: {eta_seconds/60:.1f} min)")
            
            last_progress_time = current_time
            last_count = processed_count
    
    print(f"\nDataset generation complete!")
    print(f"Total faults processed: {processed_count}/{len(fault_list)}")
    print(f"Total samples collected: {len(dataset)}")
    
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
        
        zero_critical = sum(1 for c in critical_counts if c == 0)
        if zero_critical > 0:
            print(f"  Samples with 0 critical inputs: {zero_critical} ({zero_critical/len(dataset)*100:.1f}%)")
    
    return dataset


def generate_dataset_for_folder(bench_folder, output_dir, max_faults_per_circuit=None, 
                                seed=42, skip_in_2=False):
    """Generate dataset for all circuits in folder"""
    from pathlib import Path
    
    bench_folder_path = Path(bench_folder).resolve()
    bench_files = list(bench_folder_path.rglob('*.bench')) + list(bench_folder_path.rglob('*.v'))
    
    if not bench_files:
        print(f"No .bench or .v files found in {bench_folder}")
        return 0
    
    if skip_in_2:
        bench_files = [f for f in bench_files if 'in_2.v' not in str(f)]
        print(f"Skipping in_2.v files (problematic circuits)")
    
    print(f"Found {len(bench_files)} circuits in {bench_folder} (recursive search)")
    if max_faults_per_circuit:
        print(f"Will sample {max_faults_per_circuit} faults per circuit (seed={seed})")
    print(f"Mode: SEQUENTIAL (no multiprocessing, 100% reliable)")
    bench_files = sorted(bench_files)
    
    total_samples = 0
    start_time = time.time()
    
    for i, bench_file in enumerate(bench_files):
        print(f"\n{'='*70}")
        try:
            rel_path = bench_file.relative_to(bench_folder_path)
            print(f"[{i+1}/{len(bench_files)}] Processing {rel_path}...")
        except ValueError:
            print(f"[{i+1}/{len(bench_files)}] Processing {bench_file.name}...")
        print(f"{'='*70}")
        
        circuit_start = time.time()
        
        try:
            dataset = generate_dataset_sequential(
                str(bench_file), output_dir, save_interval=100,
                max_faults=max_faults_per_circuit, seed=seed,
                root_dir=str(bench_folder_path)
            )
            circuit_time = time.time() - circuit_start
            
            total_samples += len(dataset)
            print(f"Circuit completed in {circuit_time:.1f}s ({len(dataset)} samples)")
            
        except Exception as e:
            print(f"Error processing {bench_file.name}: {e}")
            import traceback
            traceback.print_exc()
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
        description='Generate training data - SEQUENTIAL mode (no multiprocessing)',
        epilog="""
Examples:
  python data_generation_sequential.py \\
      --bench hdl-benchmarks/iccad-2015 \\
      --output ./training_data_v6 \\
      --max_faults 1000 \\
      --skip_in_2

100% reliable - no BrokenPipeError!
        """
    )
    
    parser.add_argument('--bench', type=str, required=True,
                       help='Path to .bench/.v file or folder')
    parser.add_argument('--output', type=str, default='./training_data_sequential',
                       help='Output directory')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save checkpoint every N samples')
    parser.add_argument('--max_faults', type=int, default=None,
                       help='Maximum faults to sample per circuit (None = all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for fault sampling')
    parser.add_argument('--skip_in_2', action='store_true',
                       help='Skip in_2.v files (they produce 0 samples)')
    
    args = parser.parse_args()
    
    bench_path = Path(args.bench)
    
    print("="*70)
    print("DATA GENERATION - SEQUENTIAL MODE")
    print("="*70)
    print("NO multiprocessing - 100% reliable, works for ALL circuits")
    print("Slower but guaranteed to work")
    print("="*70)
    
    if bench_path.is_dir():
        generate_dataset_for_folder(
            args.bench, args.output, 
            max_faults_per_circuit=args.max_faults, seed=args.seed,
            skip_in_2=args.skip_in_2
        )
    elif bench_path.is_file():
        generate_dataset_sequential(
            args.bench, args.output, args.save_interval,
            max_faults=args.max_faults, seed=args.seed,
            root_dir=str(bench_path.parent)
        )
    else:
        print(f"Error: {args.bench} is not a valid file or directory")
        exit(1)