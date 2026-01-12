"""
QUICK START GUIDE - GNN-Guided SAT Solver

This file demonstrates the complete workflow from scratch.
Run this after setting up the glucose solver.
"""

import os
import sys
from pathlib import Path

# Make sure you're in the DFT_SAT directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def step1_verify_dependencies():
    """Step 1: Check all dependencies are installed."""
    print("\n" + "="*70)
    print("STEP 1: Verifying Dependencies")
    print("="*70)
    
    required_packages = {
        'torch': 'PyTorch (ML framework)',
        'torch_geometric': 'PyTorch Geometric (Graph NN)',
        'pysat': 'PySAT (SAT solver library)',
    }
    
    missing = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {description}: installed")
        except ImportError:
            print(f"✗ {description}: NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def step2_build_glucose():
    """Step 2: Build Glucose C++ solver."""
    print("\n" + "="*70)
    print("STEP 2: Building Glucose SAT Solver")
    print("="*70)
    
    glucose_dir = os.path.join(os.getcwd(), "glucose", "parallel")
    
    if not os.path.exists(glucose_dir):
        print(f"✗ Glucose directory not found: {glucose_dir}")
        return False
    
    # Check if already built
    glucose_binary = os.path.join(glucose_dir, "glucose")
    if os.path.exists(glucose_binary) and os.access(glucose_binary, os.X_OK):
        print(f"✓ Glucose already built: {glucose_binary}")
        return True
    
    print(f"Building Glucose from {glucose_dir}...")
    print("Running: cd glucose/parallel && make")
    
    import subprocess
    result = subprocess.run(
        "cd glucose/parallel && make",
        shell=True,
        cwd=os.getcwd()
    )
    
    if result.returncode == 0:
        print(f"✓ Glucose built successfully")
        return True
    else:
        print(f"✗ Glucose build failed")
        return False


def step3_check_gnn_model():
    """Step 3: Check if GNN model exists."""
    print("\n" + "="*70)
    print("STEP 3: Checking GNN Model")
    print("="*70)
    
    model_file = "gnn_model_importance_aware_16feat.pth"
    
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024*1024)
        print(f"✓ GNN model found: {model_file} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"⚠ GNN model not found: {model_file}")
        print(f"\nTo train the model:")
        print(f"  python data_train_bench_mem_efficient.py")
        print(f"\n⚠ You can still use SAT solver without GNN guidance")
        return False


def step4_test_parser():
    """Step 4: Test circuit parsing."""
    print("\n" + "="*70)
    print("STEP 4: Testing Circuit Parser")
    print("="*70)
    
    from BenchParser import BenchParser
    
    # Find a test circuit
    test_circuits = list(Path(".").glob("*.bench"))
    if not test_circuits:
        print("⚠ No .bench files found in current directory")
        print("  Please add circuit files or point to a circuit directory")
        return False
    
    test_circuit = str(test_circuits[0])
    print(f"Testing parser with: {test_circuit}")
    
    try:
        parser = BenchParser(test_circuit)
        print(f"✓ Circuit parsed successfully")
        print(f"  - Inputs: {len(parser.inputs)}")
        print(f"  - Outputs: {len(parser.outputs)}")
        print(f"  - Gates: {len(parser.gates)}")
        print(f"  - DFFs: {len(parser.dffs)}")
        return True
    except Exception as e:
        print(f"✗ Parser failed: {e}")
        return False


def step5_test_gnn_solver():
    """Step 5: Test GNN-Guided SAT Solver."""
    print("\n" + "="*70)
    print("STEP 5: Testing GNN-Guided SAT Solver")
    print("="*70)
    
    if not os.path.exists("gnn_model_importance_aware_16feat.pth"):
        print("⚠ GNN model not available, skipping this step")
        return True
    
    from GNN_GUIDED_SAT_SOLVER_new import GNNGuidedSATSolver
    from pathlib import Path
    
    # Find test circuit
    test_circuits = list(Path(".").glob("*.bench"))
    if not test_circuits:
        print("⚠ No .bench files found")
        return False
    
    test_circuit = str(test_circuits[0])
    
    try:
        print(f"Initializing GNN solver...")
        solver = GNNGuidedSATSolver("gnn_model_importance_aware_16feat.pth")
        
        print(f"Running GNN-guided solve on: {Path(test_circuit).name}")
        results = solver.solve_with_gnn_guidance(
            bench_file=test_circuit,
            fault_wire=None,  # Auto-detect
            timeout=60
        )
        
        print(f"✓ Solve completed")
        print(f"  - Satisfiable: {results['satisfiable']}")
        print(f"  - Conflicts: {results['conflicts']}")
        print(f"  - Decisions: {results['decisions']}")
        return True
        
    except Exception as e:
        print(f"✗ Solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step6_run_full_pipeline():
    """Step 6: Run complete analysis pipeline."""
    print("\n" + "="*70)
    print("STEP 6: Running Full Pipeline")
    print("="*70)
    
    from CIRCUIT_SAT_INTEGRATION_new import CircuitSATAnalyzer
    from pathlib import Path
    
    # Find test circuit
    test_circuits = list(Path(".").glob("*.bench"))
    if not test_circuits:
        print("⚠ No .bench files found")
        return False
    
    test_circuit = str(test_circuits[0])
    
    try:
        print(f"Initializing analyzer...")
        analyzer = CircuitSATAnalyzer(
            gnn_model_path="gnn_model_importance_aware_16feat.pth" 
                if os.path.exists("gnn_model_importance_aware_16feat.pth") 
                else None,
            glucose_dir="./glucose"
        )
        
        print(f"Analyzing: {Path(test_circuit).name}")
        results = analyzer.analyze_circuit(
            bench_file=test_circuit,
            output_dir="example_results"
        )
        
        print(f"✓ Analysis completed")
        print(f"  - Results saved to: example_results/")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all setup steps."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "GNN-Guided SAT Solver - Quick Start" + " "*19 + "║")
    print("╚" + "="*68 + "╝")
    
    steps = [
        ("Dependencies", step1_verify_dependencies),
        ("Build Glucose", step2_build_glucose),
        ("Check GNN Model", step3_check_gnn_model),
        ("Test Parser", step4_test_parser),
        ("Test GNN Solver", step5_test_gnn_solver),
        ("Full Pipeline", step6_run_full_pipeline),
    ]
    
    results = []
    for name, func in steps:
        try:
            success = func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n✓ All setup steps completed successfully!")
        print("\nNext steps:")
        print("1. Run analysis on your circuits:")
        print("   python CIRCUIT_SAT_INTEGRATION_new.py your_circuit.bench \\")
        print("       --gnn-model gnn_model_importance_aware_16feat.pth")
        print("\n2. Run batch analysis:")
        print("   python CIRCUIT_SAT_INTEGRATION_new.py circuits_dir/ \\")
        print("       --batch \\")
        print("       --gnn-model gnn_model_importance_aware_16feat.pth \\")
        print("       --report results.json")
    else:
        print("\n✗ Some setup steps failed. Please fix the issues above.")
        print("\nFor help, see GNN_SAT_ARCHITECTURE_GUIDE_new.md")


if __name__ == "__main__":
    main()
