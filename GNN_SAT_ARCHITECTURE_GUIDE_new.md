"""
GNN-Guided SAT Solver - Architecture & Usage Guide

Overview
========
This system combines a Graph Neural Network (GNN) with the Glucose SAT solver
to efficiently solve circuit fault detection problems.

Key Innovation: Instead of SAT solving blindly, we use a trained GNN to predict
which circuit variables (wires) are most important for detecting faults, then
guide the SAT solver to branch on those variables first.


ARCHITECTURE
============

┌─────────────────────────────────────────────────────────────────┐
│                    GNN-GUIDED SAT SOLVER                        │
└─────────────────────────────────────────────────────────────────┘

1. CIRCUIT PARSING LAYER
   ├─ BenchParser.py
   │  ├─ Parses .bench files (circuit descriptions)
   │  ├─ Builds gate-level graph representation
   │  └─ Handles full-scan DFFs (pseudo inputs/outputs)
   │
   └─ Graph Features:
      ├─ Nodes: Circuit wires (inputs, gates, outputs)
      ├─ Edges: Gate connections
      └─ Node Features: Gate type, controllability, observability

2. SAT PROBLEM GENERATION
   ├─ WireFaultMiter.py
   │  ├─ Creates miter circuit (good vs. faulty)
   │  ├─ Injects stuck-at faults
   │  └─ Builds CNF (Conjunctive Normal Form) for SAT solver
   │
   └─ Output: CNF formula with ~N×M clauses
      (N = circuit gates, M = fault injection overhead)

3. GNN-GUIDED IMPORTANCE PREDICTION
   ├─ GNN_GUIDED_SAT_SOLVER_new.py::GNNImportancePredictor
   │  ├─ Input: Graph representation of circuit
   │  ├─ Architecture: GAT (Graph Attention Networks)
   │  └─ Output: Importance scores for each variable [0-1]
   │
   └─ Score Interpretation:
      └─ High score = Variable is critical for detecting fault
         (Good candidate for early branching in SAT solver)

4. SAT SOLVER INTEGRATION
   ├─ GLUCOSE_WRAPPER_new.py
   │  ├─ C++ Glucose solver interface
   │  ├─ DIMACS format conversion
   │  └─ Result parsing (conflicts, decisions, model)
   │
   └─ GNN_GUIDED_SAT_SOLVER_new.py::GNNGuidedSATSolver
      ├─ Orchestrates the full pipeline
      ├─ Passes GNN scores to SAT solver as hints
      └─ Collects solver statistics

5. INTEGRATION & ORCHESTRATION
   └─ CIRCUIT_SAT_INTEGRATION_new.py::CircuitSATAnalyzer
      ├─ Single circuit analysis
      ├─ Batch circuit analysis
      └─ Report generation


DATA FLOW DIAGRAM
=================

Circuit File (.bench)
        │
        ▼
   ┌─────────────┐
   │BenchParser  │ ──► Parses gates, connections
   └─────────────┘
        │
        ├──────────────────────┬──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
   ┌──────────┐         ┌──────────────┐      ┌─────────────┐
   │Graph Data│         │WireFaultMiter│      │GNN Features │
   │(nodes,   │         │              │      │(SCOAP, etc.)│
   │edges)    │         │Builds CNF    │      └─────────────┘
   └──────────┘         │              │            │
        │               └──────────────┘            │
        │                      │                    │
        │                      ▼                    │
        │              ┌──────────────┐            │
        │              │SAT Problem   │            │
        │              │(DIMACS CNF)  │            │
        │              └──────────────┘            │
        │                      ▲                    │
        │                      │                    │
        └──────────────────────┼────────────────────┘
                               │
                        ┌──────────────┐
                        │GNNPredictor  │
                        │              │
                        │Predict var   │
                        │importance    │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Variable      │
                        │Ranking       │
                        │[high→low]    │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │GlucoseSolver │ ◄─── Hints from GNN
                        │              │
                        │ BRANCH on    │
                        │ high-score   │
                        │ variables    │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Solution      │
                        │- Satisfiable?│
                        │- Conflicts   │
                        │- Decisions   │
                        │- Model       │
                        └──────────────┘


FILES CREATED
=============

1. GNN_GUIDED_SAT_SOLVER_new.py (Main Implementation)
   ├─ GNNGuidedSATSolver: Orchestration class
   │  ├─ load_gnn_model(): Load trained GNN
   │  ├─ extract_gnn_features(): Parse circuit to graph
   │  ├─ predict_variable_importance(): Run GNN inference
   │  ├─ build_miter_cnf(): Create SAT problem
   │  ├─ get_variable_ordering(): Rank vars by importance
   │  ├─ solve_with_gnn_guidance(): Complete pipeline
   │  └─ export_cnf_to_dimacs(): Save to DIMACS format
   │
   └─ GNNImportancePredictor: The GNN model
      ├─ GAT layers for graph feature learning
      └─ Output layer for importance prediction

2. GLUCOSE_WRAPPER_new.py (C++ Solver Interface)
   ├─ GlucoseSolverWrapper: Wrapper class
   │  ├─ solve_dimacs(): Solve DIMACS CNF file
   │  ├─ solve_from_cnf(): Solve PySAT CNF object
   │  └─ _parse_glucose_output(): Parse solver results
   │
   └─ Handles:
      ├─ Subprocess communication with C++ binary
      ├─ DIMACS format conversion
      └─ Result parsing and statistics extraction

3. CIRCUIT_SAT_INTEGRATION_new.py (Main Integration)
   ├─ CircuitSATAnalyzer: High-level orchestrator
   │  ├─ analyze_circuit(): Single circuit analysis
   │  ├─ batch_analyze(): Analyze multiple circuits
   │  └─ generate_report(): Batch reporting
   │
   └─ Usage:
      ├─ Python API
      └─ Command-line interface


USAGE EXAMPLES
==============

EXAMPLE 1: Single Circuit Analysis
───────────────────────────────────

from CIRCUIT_SAT_INTEGRATION_new import CircuitSATAnalyzer

# Initialize analyzer with trained GNN
analyzer = CircuitSATAnalyzer(
    gnn_model_path="gnn_model_importance_aware_16feat.pth",
    glucose_dir="./glucose"
)

# Analyze a circuit
results = analyzer.analyze_circuit(
    bench_file="path/to/circuit.bench",
    fault_wire="G28",          # Target wire for fault injection
    fault_type=0,              # Stuck-at-0
    output_dir="results/"
)

# Access results
print(f"Satisfiable: {results['solving_results']['gnn_guided']['satisfiable']}")
print(f"Conflicts: {results['solving_results']['gnn_guided']['conflicts']}")


EXAMPLE 2: Batch Analysis
──────────────────────────

# Analyze all circuits in a directory
results = analyzer.batch_analyze(
    bench_dir="../hdl-benchmarks/",
    output_dir="batch_results/",
    max_circuits=50
)

# Generate summary report
analyzer.generate_report(results, "summary_report.json")


EXAMPLE 3: Using GNN Solver Directly
─────────────────────────────────────

from GNN_GUIDED_SAT_SOLVER_new import GNNGuidedSATSolver

solver = GNNGuidedSATSolver("gnn_model_importance_aware_16feat.pth")

# Complete pipeline
results = solver.solve_with_gnn_guidance(
    bench_file="circuit.bench",
    fault_wire="G28",
    timeout=300
)


EXAMPLE 4: Using Glucose Wrapper Only
──────────────────────────────────────

from GLUCOSE_WRAPPER_new import GlucoseSolverWrapper
from pysat.formula import CNF

# Create CNF problem
cnf = CNF()
cnf.append([1, 2, -3])
cnf.append([-1, 2])

# Solve with Glucose
solver = GlucoseSolverWrapper()
results = solver.solve_from_cnf(cnf, timeout=300)

print(f"Satisfiable: {results['satisfiable']}")
print(f"Model: {results['model']}")


COMMAND-LINE USAGE
==================

# Single circuit with GNN guidance
python CIRCUIT_SAT_INTEGRATION_new.py \
    circuit.bench \
    --gnn-model gnn_model_importance_aware_16feat.pth \
    --output-dir results/

# Batch analysis with report
python CIRCUIT_SAT_INTEGRATION_new.py \
    ../hdl-benchmarks/ \
    --batch \
    --gnn-model gnn_model_importance_aware_16feat.pth \
    --output-dir results/ \
    --report summary_report.json \
    --max-circuits 100

# Analyze specific fault
python CIRCUIT_SAT_INTEGRATION_new.py \
    circuit.bench \
    --fault-wire G28 \
    --fault-type 0 \
    --output-dir results/


HOW IT WORKS - DETAILED EXPLANATION
====================================

1. CIRCUIT PARSING
   ────────────────
   
   Input: .bench file (circuit netlist in bench format)
   
   Example .bench content:
   INPUT(a)
   INPUT(b)
   OUTPUT(c)
   G1 = AND(a, b)
   G2 = OR(G1, a)
   c = G2
   
   Output:
   - nodes: ['a', 'b', 'G1', 'G2', 'c']
   - edges: [('a','G1'), ('b','G1'), ('G1','G2'), ('a','G2'), ('G2','c')]
   - gate_map: {'G1': ('AND', ['a','b']), ...}

2. GNN FEATURE EXTRACTION
   ───────────────────────
   
   For each wire, calculate:
   - Controllability (CC0, CC1): How hard to set to 0/1 from inputs
   - Observability (CO): How hard to observe at outputs
   - Fanin/Fanout: Number of incoming/outgoing connections
   - Gate type: AND, OR, NAND, etc.
   
   Result: 16-dimensional feature vector per node
   
   Graph: PyTorch Geometric Data object with edge_index

3. GNN IMPORTANCE PREDICTION
   ──────────────────────────
   
   Pass graph through GAT (Graph Attention Networks):
   - Attention mechanism learns which nodes influence outputs
   - Propagates information through graph
   - Outputs importance score [0,1] per variable
   
   High score = Variable appears in critical fault detection paths
   
4. MITER + FAULT INJECTION
   ────────────────────────
   
   Create dual circuit:
   
   Good circuit: Original gates and connections
   Faulty circuit: Same as good, but wire stuck-at-0/1
   
   Comparator: XOR all outputs between good and faulty
   
   CNF requirement: Comparator output = 1 (detects fault)
   
   This creates a SAT problem: Is there an input assignment that
   makes the stuck-at fault observable?

5. SAT SOLVING WITH GNN GUIDANCE
   ──────────────────────────────
   
   Traditional SAT solving:
   - Start with all variables unassigned
   - Use heuristics to pick next variable to branch on
   - Try both assignments (0, 1)
   - Recursively solve
   
   GNN-Guided SAT solving:
   - Same process BUT...
   - GNN tells solver: "Variable X is 0.95 important"
   - Solver tries X first (before less important variables)
   - Often finds solution faster with fewer decisions
   
   Result metrics:
   - conflicts: Number of dead ends
   - decisions: Number of branch points
   - propagations: Number of unit propagations
   - model: Assignment (if SAT)

EXPECTED BENEFITS
=================

1. REDUCED SOLVE TIME
   Fewer decisions needed when GNN guides toward critical variables
   
2. INTERPRETABILITY
   GNN importance scores explain why certain variables matter
   
3. FAULT DIAGNOSIS
   Ranking variables by importance shows most critical detection paths
   
4. SCALABILITY
   GNN-guided approach scales better than heuristics on large circuits


INTEGRATION WITH EXISTING CODE
===============================

Your existing files are used seamlessly:

├─ BenchParser.py (EXISTING)
│  └─ Parses .bench files
│
├─ WireFaultMiter.py (EXISTING)
│  └─ Creates CNF with fault injection
│
├─ neuro_utils.py (EXISTING)
│  └─ Provides SCOAP controllability/observability
│
└─ NEW FILES:
   ├─ GNN_GUIDED_SAT_SOLVER_new.py
   ├─ GLUCOSE_WRAPPER_new.py
   └─ CIRCUIT_SAT_INTEGRATION_new.py


NEXT STEPS
==========

1. Build Glucose C++ solver:
   cd glucose/parallel
   make

2. Train GNN model (if not already done):
   python data_train_bench_mem_efficient.py

3. Run single circuit analysis:
   python CIRCUIT_SAT_INTEGRATION_new.py circuit.bench \
       --gnn-model gnn_model_importance_aware_16feat.pth

4. Run batch analysis:
   python CIRCUIT_SAT_INTEGRATION_new.py circuits/ \
       --batch \
       --gnn-model gnn_model_importance_aware_16feat.pth \
       --report summary.json

5. Analyze results and compare with standard SAT solving


TROUBLESHOOTING
===============

Q: "Glucose solver not found"
A: Build glucose first:
   cd glucose/parallel && make
   
   Then either:
   - Use --glucose-dir flag
   - Or set GLUCOSE_DIR environment variable

Q: "GNN model not found"
A: Train the model first:
   python data_train_bench_mem_efficient.py
   
   Or provide correct path with --gnn-model

Q: "Out of memory during batch analysis"
A: Use --max-circuits to limit batch size:
   python CIRCUIT_SAT_INTEGRATION_new.py circuits/ \
       --batch --max-circuits 10

Q: "CNF generation fails"
A: Ensure:
   - .bench file is valid (test with BenchParser)
   - Circuit is synthesized (no unresolved signals)
   - Sufficient memory for large circuits (>10k gates)


REFERENCES
==========

1. SAT Solving: https://www.satcompetition.org/
2. Glucose Solver: http://www.labri.fr/perso/lsimon/research/glucose/
3. Graph Neural Networks: https://pytorch-geometric.readthedocs.io/
4. PySAT: https://pysathq.github.io/
5. Circuit Testing: DFT techniques and fault models


AUTHOR NOTES
============

This system is designed for fault detection in digital circuits using
satisfiability testing. The GNN guidance principle extends to any SAT
problem where a learnable heuristic can predict variable importance.

Key insight: GNN learns structural importance from training data, then
applies this learned importance to guide SAT solving on new circuits.
This can significantly reduce solving complexity compared to general-purpose
SAT heuristics.
"""

print(__doc__)
