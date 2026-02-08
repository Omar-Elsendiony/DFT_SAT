"""
Improved data generation with critical input identification.
This filters training data to only include inputs that actually affect fault detection.
"""

from pysat.solvers import Glucose3, Minisat22
import torch
from torch_geometric.data import Data
import multiprocessing as mp
import random
import os
import pickle

# Assuming these imports work in your environment
from BenchParser import BenchParser
from VerilogParser import VerilogParser  
from WireFaultMiter import WireFaultMiter
from neuro_utils import VectorizedGraphExtractor

CONFLICT_BUDGET = 10000
CRITICAL_INPUT_TEST_BUDGET = 50  # Quick tests for critical input detection


def identify_critical_inputs(clauses, assignment, cone_inputs, var_map):
    """
    Identify which inputs are actually critical for detecting the fault.
    
    An input is critical if flipping its polarity makes the SAT instance UNSAT
    (i.e., the fault becomes undetectable).
    
    Args:
        clauses: CNF clauses for the miter
        assignment: SAT solution (set of positive literals)
        cone_inputs: List of input signal names in the fault cone
        var_map: Dict mapping signal names to variable IDs
    
    Returns:
        dict: {input_name: polarity} for only critical inputs
              polarity is 1.0 for True, 0.0 for False
    """
    critical_inputs = {}
    
    for inp in cone_inputs:
        if inp not in var_map:
            continue
            
        var_id = var_map[inp]
        
        # What's the current polarity in the solution?
        correct_polarity = var_id in assignment
        
        # Try to solve with the OPPOSITE polarity
        test_literal = -var_id if correct_polarity else var_id
        
        with Glucose3(bootstrap_with=clauses) as probe:
            probe.conf_budget(CRITICAL_INPUT_TEST_BUDGET)
            result = probe.solve(assumptions=[test_literal])
            
            if not result:  # UNSAT
                # This input is critical! Flipping it breaks fault detection
                critical_inputs[inp] = 1.0 if correct_polarity else 0.0
    
    return critical_inputs


def process_single_fault(args):
    """
    Process a single fault with critical input identification.
    
    Returns:
        Data object or None if fault is untestable or has no critical inputs
    """
    bench_file, fault_name, fault_type, extractor_data = args
    
    try:
        # Parse circuit
        if bench_file.endswith('.bench'):
            parser = BenchParser(bench_file)
        else:
            parser = VerilogParser(bench_file)
        
        # Create fault miter
        miter = WireFaultMiter(parser)
        fault_type_name = "SA1" if fault_type == 1 else "SA0"
        clauses = miter.create_wire_fault_miter(fault_name, fault_type_name)
        
        if not clauses:
            return None
        
        # Get complete ATPG cone
        complete_cone = miter.get_complete_cone(fault_name)
        if not complete_cone:
            return None
        
        # Solve to find if fault is testable
        with Glucose3(bootstrap_with=clauses) as solver:
            solver.conf_budget(CONFLICT_BUDGET)
            if not solver.solve():
                return None  # Untestable fault
            
            assignment = set(solver.get_model())
        
        # Get inputs in the cone
        cone_inputs = miter.get_cone_inputs(complete_cone)
        if not cone_inputs:
            return None
        
        # CRITICAL STEP: Identify which inputs are actually critical
        critical_inputs = identify_critical_inputs(
            clauses, assignment, cone_inputs, miter.var_map
        )
        
        # Only create training sample if we found critical inputs
        if len(critical_inputs) < 1:  # Need at least 1 critical input
            return None
        
        # Create graph data for this fault
        # Assuming extractor is a VectorizedGraphExtractor instance
        # You'll need to reconstruct it from extractor_data
        extractor = VectorizedGraphExtractor.from_saved_data(extractor_data)
        data = extractor.get_data_for_fault(fault_name, fault_type=fault_type)
        
        # Build labels - ONLY for critical inputs
        y_polarity = torch.zeros(len(data.node_names), 1)
        train_mask = torch.zeros(len(data.node_names), 1)
        importance = torch.zeros(len(data.node_names), 1)
        
        for k, node_name in enumerate(data.node_names):
            if node_name in critical_inputs:
                y_polarity[k] = critical_inputs[node_name]
                train_mask[k] = 1.0
                importance[k] = 1.0  # All critical inputs have equal importance
        
        # Attach to data object
        data.y_polarity = y_polarity
        data.train_mask = train_mask
        data.y_importance = importance
        
        # Store metadata
        data.fault_name = fault_name
        data.fault_type = fault_type
        data.num_critical_inputs = len(critical_inputs)
        
        return data
        
    except Exception as e:
        print(f"Error processing fault {fault_name}: {e}")
        return None


def generate_dataset_parallel(bench_file, output_dir, num_workers=4):
    """
    Generate training dataset with parallel processing.
    
    Args:
        bench_file: Path to .bench or .v file
        output_dir: Where to save the dataset
        num_workers: Number of parallel workers
    """
    
    # Parse circuit once
    if bench_file.endswith('.bench'):
        parser = BenchParser(bench_file)
    else:
        parser = VerilogParser(bench_file)
    
    # Get all gates for fault injection
    all_gates = list(parser.gates.keys())
    
    # Create extractor and save its data for worker processes
    extractor = VectorizedGraphExtractor(parser)
    extractor_data = extractor.save_data()
    
    # Generate fault list (both SA0 and SA1 for each gate)
    fault_list = []
    for gate in all_gates:
        fault_list.append((bench_file, gate, 0, extractor_data))  # SA0
        fault_list.append((bench_file, gate, 1, extractor_data))  # SA1
    
    print(f"Processing {len(fault_list)} faults using {num_workers} workers...")
    
    # Process in parallel
    dataset = []
    with mp.Pool(num_workers) as pool:
        results = pool.imap_unordered(process_single_fault, fault_list, chunksize=10)
        
        for i, data in enumerate(results):
            if data is not None:
                dataset.append(data)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(fault_list)} faults, "
                      f"collected {len(dataset)} samples")
    
    print(f"\nDataset generation complete!")
    print(f"Total samples: {len(dataset)}")
    
    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    circuit_name = os.path.basename(bench_file).replace('.bench', '').replace('.v', '')
    save_path = os.path.join(output_dir, f'{circuit_name}_critical_inputs.pkl')
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Saved to {save_path}")
    
    # Print statistics
    if dataset:
        critical_counts = [d.num_critical_inputs for d in dataset]
        print(f"\nCritical input statistics:")
        print(f"  Min: {min(critical_counts)}")
        print(f"  Max: {max(critical_counts)}")
        print(f"  Avg: {sum(critical_counts) / len(critical_counts):.2f}")
    
    return dataset


def load_and_merge_datasets(data_dir):
    """
    Load and merge multiple dataset files.
    """
    dataset = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('_critical_inputs.pkl'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                dataset.extend(data)
                print(f"Loaded {len(data)} samples from {filename}")
    
    print(f"\nTotal dataset size: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training data with critical input filtering')
    parser.add_argument('--bench', type=str, required=True, help='Path to .bench or .v file')
    parser.add_argument('--output', type=str, default='./training_data_critical', 
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    generate_dataset_parallel(args.bench, args.output, args.workers)