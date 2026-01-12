"""
COMPLETE BENCHMARK PIPELINE - GNN-Guided SAT Solver

This integrates:
1. Data generation (existing: data_train_bench_mem_efficient.py)
2. GNN training (existing: data_train_bench_mem_efficient.py)
3. Benchmark comparison (existing: data_train_bench_mem_efficient.py)
4. NEW: Advanced benchmarking with our GNN_GUIDED_SAT_SOLVER
   - Compares multiple solving strategies
   - Detailed metrics (conflicts, decisions, propagations)
   - JSON report generation
"""

import os
import sys
import time
import csv
import json
import torch
import random
from pathlib import Path
from typing import Dict, List, Tuple
from pysat.solvers import Glucose3
from pysat.formula import CNF
from tqdm import tqdm

# Import existing modules
from BenchParser import BenchParser
from WireFaultMiter import WireFaultMiter
from data_train_bench_mem_efficient import (
    FastGraphExtractor,
    CircuitGNN_ImportanceAware,
    generate_importance_aware_dataset,
    train_importance_aware_model,
    BENCHMARK_DIR,
    MODEL_PATH,
    get_target_files
)

# Import new modules
from GNN_GUIDED_SAT_SOLVER_new import GNNGuidedSATSolver
from GLUCOSE_WRAPPER_new import GlucoseSolverWrapper


class AdvancedBenchmarkSuite:
    """
    Advanced benchmarking suite for GNN-Guided SAT Solver
    
    Compares:
    1. Pure Glucose (no hints)
    2. GNN-Guided Glucose (with importance hints)
    3. GNN-Guided with top-K assumptions (most aggressive)
    """
    
    def __init__(self, model_path: str = MODEL_PATH, device: str = "cpu"):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = None
        self.gnn_solver = None
        self.glucose_wrapper = None
        
        self._load_model()
        self._init_solvers()
    
    def _load_model(self):
        """Load trained GNN model."""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            print(f"   Run: python data_train_bench_mem_efficient.py train")
            return False
        
        self.model = CircuitGNN_ImportanceAware(num_node_features=16, num_layers=8)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ GNN Model loaded: {self.model_path}")
        return True
    
    def _init_solvers(self):
        """Initialize solver wrappers."""
        try:
            self.gnn_solver = GNNGuidedSATSolver(self.model_path, device=str(self.device))
            print(f"✓ GNN-Guided SAT Solver initialized")
        except Exception as e:
            print(f"⚠ GNN Solver init failed: {e}")
        
        try:
            self.glucose_wrapper = GlucoseSolverWrapper()
            print(f"✓ Glucose wrapper initialized")
        except Exception as e:
            print(f"⚠ Glucose wrapper init failed: {e}")
    
    def benchmark_circuit(self, circuit_file: str, num_faults: int = 20) -> List[Dict]:
        """
        Benchmark a single circuit with multiple faults.
        
        Args:
            circuit_file: Path to .bench circuit
            num_faults: Number of random faults to test
            
        Returns:
            List of result dictionaries
        """
        circuit_name = os.path.basename(circuit_file).replace('.bench', '')
        print(f"\n{'='*70}")
        print(f"Benchmarking: {circuit_name}")
        print(f"{'='*70}")
        
        results = []
        
        try:
            parser = BenchParser(circuit_file)
            miter = WireFaultMiter(circuit_file)
            
            if not miter.gates:
                print(f"⚠ No gates in circuit")
                return results
            
            print(f"Circuit stats: {len(parser.inputs)} inputs, {len(parser.outputs)} outputs, {len(parser.gates)} gates")
            
            # Select random faults
            fault_targets = random.sample(
                [g[0] for g in miter.gates],
                min(num_faults, len(miter.gates))
            )
            
            for i, fault_wire in enumerate(tqdm(fault_targets, desc="Testing faults")):
                result = self._solve_with_all_methods(
                    circuit_file, fault_wire, circuit_name, i+1, num_faults
                )
                results.append(result)
        
        except Exception as e:
            print(f"❌ Error benchmarking {circuit_name}: {e}")
        
        return results
    
    def _solve_with_all_methods(self, circuit_file: str, fault_wire: str, 
                               circuit_name: str, fault_num: int, 
                               total_faults: int) -> Dict:
        """
        Solve same SAT problem with 3 different methods.
        
        Methods:
        1. Pure Glucose (baseline)
        2. GNN-Guided Glucose (with variable ordering hints)
        3. GNN Top-K Assumptions (most aggressive)
        """
        
        result = {
            'circuit': circuit_name,
            'fault_wire': fault_wire,
            'fault_num': fault_num,
            'total_faults': total_faults,
        }
        
        # Build CNF
        try:
            miter = WireFaultMiter(circuit_file)
            clauses = miter.build_miter(fault_wire, fault_type=0)
            cnf = CNF(from_clauses=clauses)
            
            result['num_clauses'] = len(clauses)
            result['num_variables'] = len(miter.var_map)
        except Exception as e:
            result['error'] = str(e)
            return result
        
        # METHOD 1: Pure Glucose (Baseline)
        print(f"  [{fault_num}/{total_faults}] {fault_wire}")
        result.update(self._solve_pure_glucose(cnf))
        
        # METHOD 2: GNN-Guided (variable ordering)
        if self.gnn_solver:
            result.update(self._solve_gnn_guided(circuit_file, fault_wire, cnf))
        
        # METHOD 3: GNN Top-K Assumptions
        if self.model:
            result.update(self._solve_gnn_topk(circuit_file, fault_wire, miter, cnf))
        
        # Calculate speedups
        if result.get('pure_conflicts') and result.get('gnn_conflicts'):
            result['speedup_conflicts'] = result['pure_conflicts'] / max(result['gnn_conflicts'], 1)
            result['speedup_time'] = result['pure_time'] / max(result['gnn_time'], 0.0001)
        
        return result
    
    def _solve_pure_glucose(self, cnf: CNF) -> Dict:
        """Standard Glucose without hints (baseline)."""
        result = {'method': 'pure_glucose'}
        
        try:
            start = time.time()
            solver = Glucose3(bootstrap_with=cnf.clauses)
            satisfiable = solver.solve()
            elapsed = time.time() - start
            
            # Try to get stats
            try:
                conflicts = solver.nof_conflicts()
                decisions = solver.nof_decisions()
                propagations = solver.nof_propagations()
            except:
                conflicts = decisions = propagations = None
            
            result.update({
                'pure_satisfiable': satisfiable,
                'pure_conflicts': conflicts or 0,
                'pure_decisions': decisions or 0,
                'pure_propagations': propagations or 0,
                'pure_time': elapsed,
            })
        except Exception as e:
            result['pure_error'] = str(e)
        
        return result
    
    def _solve_gnn_guided(self, circuit_file: str, fault_wire: str, cnf: CNF) -> Dict:
        """GNN-Guided solving (variable ordering hints)."""
        result = {}
        
        if not self.gnn_solver:
            return result
        
        try:
            start = time.time()
            
            # Extract features
            data, idx_to_name = self.gnn_solver.extract_gnn_features(circuit_file)
            
            # Predict importance
            importance_scores = self.gnn_solver.predict_variable_importance(data)
            
            # Get variable ordering
            miter = WireFaultMiter(circuit_file)
            var_map = miter.var_map
            gnn_var_ordering = self.gnn_solver.get_variable_ordering(
                importance_scores, idx_to_name, var_map
            )
            
            # Solve with hints
            solver = Glucose3(bootstrap_with=cnf.clauses)
            if gnn_var_ordering:
                solver.set_phases(gnn_var_ordering)
            
            satisfiable = solver.solve()
            elapsed = time.time() - start
            
            # Get stats
            try:
                conflicts = solver.nof_conflicts()
                decisions = solver.nof_decisions()
                propagations = solver.nof_propagations()
            except:
                conflicts = decisions = propagations = None
            
            result.update({
                'gnn_satisfiable': satisfiable,
                'gnn_conflicts': conflicts or 0,
                'gnn_decisions': decisions or 0,
                'gnn_propagations': propagations or 0,
                'gnn_time': elapsed,
                'gnn_var_hints': len(gnn_var_ordering),
            })
        except Exception as e:
            result['gnn_error'] = str(e)
        
        return result
    
    def _solve_gnn_topk(self, circuit_file: str, fault_wire: str, 
                        miter: WireFaultMiter, cnf: CNF) -> Dict:
        """GNN Top-K assumptions (most aggressive)."""
        result = {}
        
        try:
            extractor = FastGraphExtractor(circuit_file, miter.var_map)
            data = extractor.get_data_for_fault(fault_wire)
            
            if data is None:
                return result
            
            data = data.to(self.device)
            
            start = time.time()
            
            with torch.no_grad():
                value_logits, importance_scores = self.model(data)
                value_probs = torch.sigmoid(value_logits)
            
            # Create assumptions from top-K important variables
            hints = []
            input_names = set(miter.inputs)
            
            for idx, name in enumerate(data.node_names if hasattr(data, 'node_names') else []):
                if name in input_names:
                    var_id = miter.var_map.get(name)
                    if var_id:
                        prob = value_probs[idx].item()
                        importance = importance_scores[idx].item()
                        hints.append((var_id, prob, importance))
            
            # Sort by importance and take top-K
            hints.sort(key=lambda x: x[2], reverse=True)
            TOP_K = 5
            assumptions = [h[0] if h[1] > 0.5 else -h[0] for h in hints[:TOP_K]]
            
            # Solve with assumptions
            solver = Glucose3(bootstrap_with=cnf.clauses)
            satisfiable = solver.solve(assumptions=assumptions)
            
            elapsed = time.time() - start
            
            try:
                conflicts = solver.nof_conflicts()
                decisions = solver.nof_decisions()
                propagations = solver.nof_propagations()
            except:
                conflicts = decisions = propagations = None
            
            result.update({
                'topk_satisfiable': satisfiable,
                'topk_conflicts': conflicts or 0,
                'topk_decisions': decisions or 0,
                'topk_propagations': propagations or 0,
                'topk_time': elapsed,
                'topk_assumptions': len(assumptions),
            })
        except Exception as e:
            result['topk_error'] = str(e)
        
        return result
    
    def run_full_benchmark(self, num_circuits: int = 10, 
                          faults_per_circuit: int = 20,
                          output_file: str = "benchmark_results_new.json") -> Dict:
        """
        Run full benchmark on multiple circuits.
        
        Args:
            num_circuits: Maximum circuits to benchmark
            faults_per_circuit: Faults per circuit
            output_file: Output JSON file
            
        Returns:
            Summary statistics
        """
        print(f"\n{'='*70}")
        print(f"GNN-GUIDED SAT SOLVER - ADVANCED BENCHMARK")
        print(f"{'='*70}")
        print(f"Benchmarking up to {num_circuits} circuits")
        print(f"Faults per circuit: {faults_per_circuit}")
        print(f"Output: {output_file}\n")
        
        all_results = []
        
        # Get circuit files
        if os.path.exists(BENCHMARK_DIR):
            circuit_files = list(Path(BENCHMARK_DIR).glob("*.bench"))[:num_circuits]
        else:
            print(f"⚠ Benchmark directory not found: {BENCHMARK_DIR}")
            print(f"  Place .bench files there or update BENCHMARK_DIR")
            return {}
        
        if not circuit_files:
            print(f"❌ No .bench files found in {BENCHMARK_DIR}")
            return {}
        
        # Benchmark each circuit
        for circuit_file in circuit_files:
            results = self.benchmark_circuit(str(circuit_file), faults_per_circuit)
            all_results.extend(results)
        
        # Generate summary
        summary = self._generate_summary(all_results)
        
        # Save results
        self._save_results(all_results, summary, output_file)
        
        return summary
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics."""
        if not results:
            return {}
        
        # Collect metrics
        pure_conflicts = [r.get('pure_conflicts', 0) for r in results if r.get('pure_conflicts')]
        gnn_conflicts = [r.get('gnn_conflicts', 0) for r in results if r.get('gnn_conflicts')]
        topk_conflicts = [r.get('topk_conflicts', 0) for r in results if r.get('topk_conflicts')]
        
        pure_times = [r.get('pure_time', 0) for r in results if r.get('pure_time')]
        gnn_times = [r.get('gnn_time', 0) for r in results if r.get('gnn_time')]
        topk_times = [r.get('topk_time', 0) for r in results if r.get('topk_time')]
        
        speedups = [r.get('speedup_conflicts', 1) for r in results if r.get('speedup_conflicts')]
        
        summary = {
            'total_runs': len(results),
            'pure_glucose': {
                'avg_conflicts': sum(pure_conflicts) / len(pure_conflicts) if pure_conflicts else 0,
                'avg_time': sum(pure_times) / len(pure_times) if pure_times else 0,
            },
            'gnn_guided': {
                'avg_conflicts': sum(gnn_conflicts) / len(gnn_conflicts) if gnn_conflicts else 0,
                'avg_time': sum(gnn_times) / len(gnn_times) if gnn_times else 0,
            },
            'gnn_topk': {
                'avg_conflicts': sum(topk_conflicts) / len(topk_conflicts) if topk_conflicts else 0,
                'avg_time': sum(topk_times) / len(topk_times) if topk_times else 0,
            },
            'avg_speedup': sum(speedups) / len(speedups) if speedups else 1.0,
        }
        
        return summary
    
    def _save_results(self, results: List[Dict], summary: Dict, output_file: str):
        """Save results to files."""
        os.makedirs('benchmark_results', exist_ok=True)
        
        # JSON results
        json_path = os.path.join('benchmark_results', output_file)
        with open(json_path, 'w') as f:
            json.dump({
                'summary': summary,
                'detailed_results': results
            }, f, indent=2)
        print(f"\n✓ JSON results saved: {json_path}")
        
        # CSV results
        csv_path = os.path.join('benchmark_results', output_file.replace('.json', '.csv'))
        if results:
            with open(csv_path, 'w', newline='') as f:
                fieldnames = list(results[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"✓ CSV results saved: {csv_path}")
        
        # Summary
        print(f"\n{'='*70}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"Total runs: {summary.get('total_runs', 0)}")
        print(f"\nAverage Conflicts:")
        print(f"  Pure Glucose:     {summary.get('pure_glucose', {}).get('avg_conflicts', 0):.0f}")
        print(f"  GNN-Guided:       {summary.get('gnn_guided', {}).get('avg_conflicts', 0):.0f}")
        print(f"  GNN Top-K:        {summary.get('gnn_topk', {}).get('avg_conflicts', 0):.0f}")
        print(f"\nAverage CPU Time:")
        print(f"  Pure Glucose:     {summary.get('pure_glucose', {}).get('avg_time', 0):.4f}s")
        print(f"  GNN-Guided:       {summary.get('gnn_guided', {}).get('avg_time', 0):.4f}s")
        print(f"  GNN Top-K:        {summary.get('gnn_topk', {}).get('avg_time', 0):.4f}s")
        print(f"\nAverage Speedup: {summary.get('avg_speedup', 1.0):.2f}x")
        print(f"{'='*70}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete GNN-Guided SAT Solver Benchmark Pipeline"
    )
    parser.add_argument('command', nargs='?', default='help',
                       choices=['generate', 'train', 'benchmark', 'advanced-bench', 'help'],
                       help='Command to run')
    parser.add_argument('--num-circuits', type=int, default=10,
                       help='Number of circuits for advanced benchmark')
    parser.add_argument('--faults-per', type=int, default=20,
                       help='Number of faults per circuit')
    parser.add_argument('--output', default='benchmark_results_new.json',
                       help='Output file name')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        print("Generating dataset...")
        generate_importance_aware_dataset()
    
    elif args.command == 'train':
        print("Training GNN model...")
        train_importance_aware_model()
    
    elif args.command == 'benchmark':
        print("Running standard benchmark...")
        from data_train_bench_mem_efficient import run_importance_guided_benchmark
        run_importance_guided_benchmark()
    
    elif args.command == 'advanced-bench':
        print("Running advanced benchmark...")
        suite = AdvancedBenchmarkSuite()
        suite.run_full_benchmark(
            num_circuits=args.num_circuits,
            faults_per_circuit=args.faults_per,
            output_file=args.output
        )
    
    else:
        print("""
Usage:
  python BENCHMARK_PIPELINE_new.py generate        # Generate dataset
  python BENCHMARK_PIPELINE_new.py train           # Train GNN model
  python BENCHMARK_PIPELINE_new.py benchmark       # Run standard benchmark
  python BENCHMARK_PIPELINE_new.py advanced-bench  # Run advanced benchmark
  
Advanced benchmark options:
  --num-circuits NUM    : Number of circuits to benchmark (default: 10)
  --faults-per NUM      : Faults per circuit (default: 20)
  --output FILE         : Output JSON file (default: benchmark_results_new.json)

Example:
  python BENCHMARK_PIPELINE_new.py advanced-bench --num-circuits 50 --faults-per 30
        """)


if __name__ == "__main__":
    main()
