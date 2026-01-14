"""
Complete Integration: GNN-Guided SAT Solver Pipeline

This script demonstrates the full workflow:
1. Train or load a GNN model for importance prediction
2. Parse circuit in .bench format
3. Inject faults and create SAT problem (CNF)
4. Use GNN to guide variable ordering
5. Solve with Glucose SAT solver
6. Analyze and report results

FIXED: Added device parameter support and proper wrapper usage
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import torch
from pysat.formula import CNF

# Import your modules
from BenchParser import BenchParser
from WireFaultMiter import WireFaultMiter
from GNN_GUIDED_SAT_SOLVER_new import GNNGuidedSATSolver

# Try to import wrapper
try:
    from GLUCOSE_WRAPPER_new import GlucoseSolverWrapper
    WRAPPER_AVAILABLE = True
except ImportError:
    WRAPPER_AVAILABLE = False


class CircuitSATAnalyzer:
    """
    Main orchestrator for GNN-guided SAT solving of circuits.
    """
    
    def __init__(self, gnn_model_path: Optional[str] = None, 
                 glucose_dir: Optional[str] = None,
                 device: str = None):
        """
        Initialize the analyzer.
        
        Args:
            gnn_model_path: Path to trained GNN model
            glucose_dir: Path to glucose solver directory
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.gnn_solver = None
        self.glucose_wrapper = None
        
        print(f"✓ Using device: {self.device}")
        
        # Load GNN if path provided
        if gnn_model_path and os.path.exists(gnn_model_path):
            print(f"Loading GNN model from {gnn_model_path}...")
            self.gnn_solver = GNNGuidedSATSolver(gnn_model_path, device)
        else:
            print("⚠ GNN model not provided or not found.")
            print("  Falling back to standard SAT solving without GNN guidance.")
        
        # Initialize Glucose wrapper
        if WRAPPER_AVAILABLE:
            try:
                self.glucose_wrapper = GlucoseSolverWrapper(glucose_dir)
                print("✓ GlucoseSolverWrapper initialized")
            except Exception as e:
                print(f"⚠ Glucose wrapper init failed: {e}")
        else:
            print("⚠ GlucoseSolverWrapper not available")
    
    def analyze_circuit(self, bench_file: str, 
                       fault_wire: Optional[str] = None,
                       fault_type: int = 0,
                       output_dir: Optional[str] = None) -> Dict:
        """
        Complete analysis pipeline for a circuit.
        
        Args:
            bench_file: Path to .bench file
            fault_wire: Wire to inject fault (if None, use automatic detection)
            fault_type: 0 (stuck-at-0) or 1 (stuck-at-1)
            output_dir: Directory to save results
            
        Returns:
            Results dictionary
        """
        
        print(f"\n{'='*70}")
        print(f"Circuit SAT Analysis Pipeline")
        print(f"{'='*70}")
        print(f"Circuit: {os.path.basename(bench_file)}")
        
        # Parse circuit
        print("\n[Step 1] Parsing circuit...")
        parser = BenchParser(bench_file)
        print(f"  ✓ Inputs: {len(parser.inputs)}")
        print(f"  ✓ Outputs: {len(parser.outputs)}")
        print(f"  ✓ Gates: {len(parser.gates)}")
        print(f"  ✓ DFFs: {len(parser.dffs)}")
        
        # Auto-detect fault wire if not specified
        if fault_wire is None:
            # Use the first gate as fault target
            if parser.gates:
                fault_wire = parser.gates[0][0]
                print(f"\n[Step 2] Auto-detected fault wire: {fault_wire}")
            else:
                raise ValueError("Circuit has no gates to inject faults")
        else:
            print(f"\n[Step 2] Injecting fault at: {fault_wire} (stuck-at-{fault_type})")
        
        # Build CNF
        print("\n[Step 3] Building CNF with fault injection...")
        miter = WireFaultMiter(bench_file)
        clauses = miter.build_miter(fault_wire, fault_type=fault_type)
        cnf = CNF(from_clauses=clauses)
        print(f"  ✓ Clauses: {len(clauses)}")
        print(f"  ✓ Variables: {len(miter.var_map)}")
        
        results = {
            'circuit_file': bench_file,
            'circuit_name': os.path.basename(bench_file).replace('.bench', ''),
            'fault_wire': fault_wire,
            'fault_type': fault_type,
            'circuit_stats': {
                'num_inputs': len(parser.inputs),
                'num_outputs': len(parser.outputs),
                'num_gates': len(parser.gates),
                'num_dffs': len(parser.dffs),
            },
            'sat_problem': {
                'num_clauses': len(clauses),
                'num_variables': len(miter.var_map),
            },
            'solving_results': {}
        }
        
        # GNN-Guided Solving (if model available)
        if self.gnn_solver:
            print("\n[Step 4] GNN-Guided SAT Solving...")
            try:
                gnn_results = self.gnn_solver.solve_with_gnn_guidance(
                    bench_file=bench_file,
                    fault_wire=fault_wire,
                    fault_type=fault_type,
                    timeout=300
                )
                results['solving_results']['gnn_guided'] = gnn_results
                print(f"  ✓ GNN-Guided Solve: {gnn_results['satisfiable']}")
                print(f"    - Conflicts: {gnn_results['conflicts']}")
                print(f"    - Decisions: {gnn_results['decisions']}")
            except Exception as e:
                print(f"  ✗ GNN solving failed: {e}")
                results['solving_results']['gnn_guided'] = {'error': str(e)}
        
        # Standard Glucose Solving (if wrapper available)
        if self.glucose_wrapper:
            print("\n[Step 5] Standard Glucose Solving...")
            try:
                glucose_results = self.glucose_wrapper.solve_from_cnf(
                    cnf,
                    timeout=300,
                    verbose=False
                )
                results['solving_results']['glucose_standard'] = glucose_results
                print(f"  ✓ Glucose Solve: {glucose_results['satisfiable']}")
                if glucose_results.get('conflicts'):
                    print(f"    - Conflicts: {glucose_results['conflicts']}")
                if glucose_results.get('decisions'):
                    print(f"    - Decisions: {glucose_results['decisions']}")
            except Exception as e:
                print(f"  ✗ Glucose solving failed: {e}")
                results['solving_results']['glucose_standard'] = {'error': str(e)}
        
        # Save results
        if output_dir:
            self._save_results(results, output_dir)
        
        return results
    
    def batch_analyze(self, bench_dir: str, 
                     output_dir: Optional[str] = None,
                     max_circuits: Optional[int] = None) -> List[Dict]:
        """
        Analyze multiple circuits in a directory.
        
        Args:
            bench_dir: Directory containing .bench files
            output_dir: Directory to save results
            max_circuits: Maximum number of circuits to analyze
            
        Returns:
            List of results dictionaries
        """
        bench_files = list(Path(bench_dir).glob("*.bench"))
        
        if max_circuits:
            bench_files = bench_files[:max_circuits]
        
        print(f"\nBatch analyzing {len(bench_files)} circuits...")
        
        all_results = []
        for i, bench_file in enumerate(bench_files, 1):
            print(f"\n[{i}/{len(bench_files)}] Processing {bench_file.name}...")
            try:
                results = self.analyze_circuit(
                    str(bench_file),
                    output_dir=output_dir
                )
                all_results.append(results)
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        return all_results
    
    def _save_results(self, results: Dict, output_dir: str):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(
            output_dir,
            f"{results['circuit_name']}_gnn_sat_results.json"
        )
        
        # Convert tensors to serializable format
        results_serialized = self._make_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(results_serialized, f, indent=2)
        
        print(f"  ✓ Results saved to {output_file}")
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def generate_report(self, results_list: List[Dict], output_file: str):
        """
        Generate summary report from multiple results.
        
        Args:
            results_list: List of results dictionaries
            output_file: Output report file path
        """
        print(f"\nGenerating report for {len(results_list)} circuits...")
        
        summary = {
            'total_circuits': len(results_list),
            'circuits': []
        }
        
        for results in results_list:
            circuit_summary = {
                'name': results['circuit_name'],
                'gates': results['circuit_stats']['num_gates'],
                'sat_clauses': results['sat_problem']['num_clauses'],
                'sat_variables': results['sat_problem']['num_variables'],
                'results': {}
            }
            
            for solver_name, solver_result in results['solving_results'].items():
                if 'error' not in solver_result:
                    circuit_summary['results'][solver_name] = {
                        'satisfiable': solver_result.get('satisfiable'),
                        'conflicts': solver_result.get('conflicts'),
                        'decisions': solver_result.get('decisions'),
                        'propagations': solver_result.get('propagations')
                    }
            
            summary['circuits'].append(circuit_summary)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Report saved to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GNN-Guided SAT Solver for Circuit Analysis"
    )
    parser.add_argument('circuit', help='Path to .bench circuit file or directory')
    parser.add_argument('--gnn-model', help='Path to trained GNN model')
    parser.add_argument('--glucose-dir', help='Path to glucose directory')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use: cpu, cuda, or auto (default: auto)')
    parser.add_argument('--fault-wire', help='Wire to inject fault')
    parser.add_argument('--fault-type', type=int, default=0, help='Fault type: 0 or 1')
    parser.add_argument('--output-dir', help='Directory to save results')
    parser.add_argument('--batch', action='store_true', help='Analyze all circuits in directory')
    parser.add_argument('--max-circuits', type=int, help='Max circuits to analyze in batch mode')
    parser.add_argument('--report', help='Generate summary report')
    
    args = parser.parse_args()
    
    # Handle device argument
    device = None if args.device == 'auto' else args.device
    
    # Initialize analyzer
    analyzer = CircuitSATAnalyzer(
        gnn_model_path=args.gnn_model,
        glucose_dir=args.glucose_dir,
        device=device
    )
    
    # Single circuit or batch analysis
    if args.batch and os.path.isdir(args.circuit):
        results = analyzer.batch_analyze(
            args.circuit,
            output_dir=args.output_dir,
            max_circuits=args.max_circuits
        )
        
        if args.report:
            analyzer.generate_report(results, args.report)
    else:
        results = analyzer.analyze_circuit(
            args.circuit,
            fault_wire=args.fault_wire,
            fault_type=args.fault_type,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(json.dumps(results['solving_results'], indent=2))


if __name__ == "__main__":
    main()