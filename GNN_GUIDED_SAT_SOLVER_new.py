"""
GNN-Guided SAT Solver for Circuit Analysis

This module combines a trained GNN model with the GluCose SAT solver to:
1. Use GNN to predict the importance/observability of variables
2. Guide the SAT solver's variable ordering (first branch on important variables)
3. Solve circuits represented in CNF format
4. Track and analyze the solution process

The GNN predicts which variables (circuit wires) are most critical for detecting faults,
and this information is passed to the SAT solver as branching hints.
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from pysat.formula import CNF
from pysat.solvers import Glucose3
import networkx as nx
from BenchParser import BenchParser
from WireFaultMiter import WireFaultMiter


class GNNGuidedSATSolver:
    """
    Main class that orchestrates GNN-guided SAT solving.
    
    Workflow:
    1. Load GNN model trained to predict variable importance
    2. Parse circuit in .bench format
    3. Build CNF formula (SAT problem)
    4. Use GNN to rank variables by importance
    5. Solve with Glucose3, respecting GNN hints
    6. Analyze solution
    """
    
    def __init__(self, gnn_model_path: str, device: str = "cpu"):
        """
        Initialize GNN-Guided SAT Solver.
        
        Args:
            gnn_model_path: Path to trained GNN model (.pth file)
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.gnn_model = None
        self.gnn_model_path = gnn_model_path
        self.load_gnn_model()
        
    def load_gnn_model(self):
        """Load the trained GNN model for importance prediction."""
        if not os.path.exists(self.gnn_model_path):
            raise FileNotFoundError(f"GNN model not found: {self.gnn_model_path}")
        
        # Define model architecture (must match training)
        # Adjust input_dim and hidden_dim based on your training config
        model = GNNImportancePredictor(
            input_dim=16,  # 16 features from FastGraphExtractor
            hidden_dim=64,
            output_dim=1,  # Predict importance score
            num_layers=2
        )
        
        checkpoint = torch.load(self.gnn_model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        self.gnn_model = model
        print(f"✓ GNN Model loaded from {self.gnn_model_path}")
    
    def extract_gnn_features(self, bench_file: str) -> Tuple[Data, Dict[int, str]]:
        """
        Extract graph features from circuit using FastGraphExtractor.
        
        Args:
            bench_file: Path to .bench circuit file
            
        Returns:
            Tuple of (PyTorch Geometric Data object, variable mapping)
        """
        from data_train_bench_mem_efficient import FastGraphExtractor
        
        extractor = FastGraphExtractor(bench_file)
        
        # Build graph data
        edge_index = extractor.edge_index
        x = extractor.x_base  # Node features
        
        # Create node name to index mapping
        name_to_idx = extractor.name_to_idx
        idx_to_name = {idx: name for name, idx in name_to_idx.items()}
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            num_nodes=extractor.num_nodes
        )
        
        print(f"✓ Extracted graph: {data.num_nodes} nodes, {data.num_edges} edges")
        return data, idx_to_name
    
    def predict_variable_importance(self, data: Data) -> torch.Tensor:
        """
        Use GNN to predict importance scores for all variables.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Tensor of importance scores [num_nodes]
        """
        data = data.to(self.device)
        
        with torch.no_grad():
            importance_scores = self.gnn_model(data.x, data.edge_index)
        
        return importance_scores.cpu()
    
    def build_miter_cnf(self, bench_file: str, fault_wire: str, fault_type: int = 0) -> Tuple[CNF, Dict]:
        """
        Build CNF formula from circuit with fault injection.
        
        Args:
            bench_file: Path to .bench circuit file
            fault_wire: Wire to inject fault on
            fault_type: 0 (stuck-at-0) or 1 (stuck-at-1)
            
        Returns:
            Tuple of (CNF formula, metadata dict with var_map)
        """
        miter = WireFaultMiter(bench_file)
        
        # Build miter clauses
        clauses = miter.build_miter(fault_wire, fault_type=fault_type)
        
        # Create CNF from clauses
        cnf = CNF(from_clauses=clauses)
        
        metadata = {
            'var_map': miter.var_map,
            'fault_wire': fault_wire,
            'fault_type': fault_type,
            'num_clauses': len(clauses),
            'num_vars': len(miter.var_map)
        }
        
        print(f"✓ Built CNF: {len(clauses)} clauses, {len(miter.var_map)} variables")
        return cnf, metadata
    
    def get_variable_ordering(self, importance_scores: torch.Tensor, 
                             idx_to_name: Dict[int, str], 
                             var_map: Dict[str, int]) -> List[int]:
        """
        Create SAT variable ordering based on GNN importance predictions.
        
        Args:
            importance_scores: GNN-predicted importance for each variable
            idx_to_name: Mapping from GNN node index to wire name
            var_map: Mapping from wire name to SAT variable ID
            
        Returns:
            List of SAT variable IDs ordered by importance (descending)
        """
        # Sort nodes by importance (descending)
        sorted_indices = torch.argsort(importance_scores.squeeze(), descending=True)
        
        var_ordering = []
        for idx in sorted_indices:
            idx_int = int(idx.item())
            if idx_int in idx_to_name:
                wire_name = idx_to_name[idx_int]
                if wire_name in var_map:
                    var_id = var_map[wire_name]
                    var_ordering.append(var_id)
        
        return var_ordering
    
    def solve_with_gnn_guidance(self, bench_file: str, fault_wire: str, 
                               fault_type: int = 0, 
                               timeout: int = 300) -> Dict:
        """
        Complete workflow: Extract features -> Predict importance -> Solve SAT.
        
        Args:
            bench_file: Path to .bench circuit file
            fault_wire: Wire to inject fault
            fault_type: 0 or 1 for stuck-at fault
            timeout: SAT solver timeout in seconds
            
        Returns:
            Results dictionary with solution, conflicts, decisions, etc.
        """
        print(f"\n{'='*70}")
        print(f"GNN-Guided SAT Solving for {os.path.basename(bench_file)}")
        print(f"Fault: {fault_wire} (stuck-at-{fault_type})")
        print(f"{'='*70}\n")
        
        # Step 1: Extract GNN features
        print("[1/4] Extracting GNN features...")
        data, idx_to_name = self.extract_gnn_features(bench_file)
        
        # Step 2: Predict importance with GNN
        print("[2/4] Predicting variable importance with GNN...")
        importance_scores = self.predict_variable_importance(data)
        
        # Step 3: Build CNF with fault
        print("[3/4] Building CNF formula with fault injection...")
        cnf, metadata = self.build_miter_cnf(bench_file, fault_wire, fault_type)
        var_map = metadata['var_map']
        
        # Step 4: Create variable ordering from GNN
        print("[4/4] Creating GNN-guided variable ordering...")
        gnn_var_ordering = self.get_variable_ordering(importance_scores, idx_to_name, var_map)
        
        # Solve with Glucose3
        print(f"\nSolving with Glucose3 SAT solver (timeout={timeout}s)...")
        solver = Glucose3(bootstrap_with=cnf.clauses)
        solver.set_phases([i for i in gnn_var_ordering])  # Set phase for hints
        
        # Solve
        satisfiable = solver.solve(assumptions=[], time_limit=timeout)
        
        # Collect results
        results = {
            'satisfiable': satisfiable,
            'bench_file': bench_file,
            'fault_wire': fault_wire,
            'fault_type': fault_type,
            'num_clauses': metadata['num_clauses'],
            'num_variables': metadata['num_vars'],
            'conflicts': solver.nof_conflicts(),
            'decisions': solver.nof_decisions(),
            'propagations': solver.nof_propagations(),
            'gnn_var_ordering_size': len(gnn_var_ordering),
            'model': solver.get_model() if satisfiable else None
        }
        
        print(f"\n✓ Solve complete!")
        print(f"  Satisfiable: {satisfiable}")
        print(f"  Conflicts: {results['conflicts']}")
        print(f"  Decisions: {results['decisions']}")
        print(f"  Propagations: {results['propagations']}")
        
        return results
    
    def export_cnf_to_dimacs(self, cnf: CNF, output_file: str):
        """
        Export CNF to DIMACS format for use with Glucose C++ solver.
        
        Args:
            cnf: CNF formula
            output_file: Output .cnf file path
        """
        cnf.to_file(output_file)
        print(f"✓ Exported CNF to DIMACS format: {output_file}")
    
    def solve_with_glucose_cpp(self, dimacs_file: str, glucose_path: str = None) -> Dict:
        """
        Solve using the C++ Glucose solver directly.
        
        Args:
            dimacs_file: DIMACS format .cnf file
            glucose_path: Path to glucose binary (auto-detected if None)
            
        Returns:
            Results dictionary with solver output
        """
        # Auto-detect glucose binary
        if glucose_path is None:
            glucose_dir = os.path.join(os.path.dirname(__file__), "glucose", "parallel")
            glucose_path = os.path.join(glucose_dir, "glucose")
            
            if not os.path.exists(glucose_path):
                raise FileNotFoundError(
                    f"Glucose solver not found at {glucose_path}. "
                    f"Please build glucose or specify path."
                )
        
        print(f"Running Glucose C++ solver...")
        print(f"Input: {dimacs_file}")
        
        try:
            result = subprocess.run(
                [glucose_path, dimacs_file],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            output = result.stdout
            print(output)
            
            return {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'satisfiable': 'SATISFIABLE' in result.stdout or result.returncode == 10
            }
            
        except subprocess.TimeoutExpired:
            return {
                'return_code': -1,
                'satisfiable': None,
                'error': 'Timeout'
            }
        except Exception as e:
            return {
                'return_code': -1,
                'satisfiable': None,
                'error': str(e)
            }


class GNNImportancePredictor(nn.Module):
    """
    GNN model for predicting variable importance scores.
    
    Architecture: GAT-based graph neural network with attention mechanism.
    Output: Single importance score per variable (0-1 range after sigmoid).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # GAT layers for feature propagation
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATv2Conv(input_dim, hidden_dim, heads=4, dropout=0.1))
        
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATv2Conv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.1))
        
        # Output layer: hidden -> importance score
        self.output_layer = nn.Linear(hidden_dim * 4, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, edge_index):
        """
        Forward pass through GNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge list [2, num_edges]
            
        Returns:
            Importance scores [num_nodes, output_dim]
        """
        # Propagate through GAT layers
        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = torch.relu(x)
        
        # Predict importance
        importance = self.output_layer(x)
        importance = self.sigmoid(importance)
        
        return importance


def example_usage():
    """Example usage of GNN-Guided SAT Solver."""
    
    # Initialize solver with trained GNN model
    solver = GNNGuidedSATSolver(
        gnn_model_path="gnn_model_importance_aware_16feat.pth",
        device="cpu"
    )
    
    # Solve a circuit with GNN guidance
    results = solver.solve_with_gnn_guidance(
        bench_file="path/to/circuit.bench",
        fault_wire="G28",
        fault_type=0,  # Stuck-at-0
        timeout=300
    )
    
    print("\nResults:")
    print(f"Satisfiable: {results['satisfiable']}")
    print(f"Conflicts: {results['conflicts']}")
    print(f"Decisions: {results['decisions']}")
    
    # Optional: Export to DIMACS and use C++ Glucose
    # solver.export_cnf_to_dimacs(cnf, "circuit.cnf")
    # results_cpp = solver.solve_with_glucose_cpp("circuit.cnf")


if __name__ == "__main__":
    example_usage()
