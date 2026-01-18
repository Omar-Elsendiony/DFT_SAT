"""
Main Pipeline for GNN-Guided SAT Solving

Usage:
    python main.py generate  # Generate training data
    python main.py train     # Train the model
    python main.py benchmark # Run benchmarks
    python main.py all       # Run complete pipeline
"""

import sys
import os

def print_usage():
    print(__doc__)

def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "generate":
        print("Starting data generation...")
        from data_generation_hybrid import generate_dataset
        generate_dataset()
        
    elif command == "train":
        print("Starting model training...")
        from train_model import train_model
        train_model()
        
    elif command == "benchmark":
        print("Starting benchmark...")
        from benchmark import run_benchmark
        run_benchmark()
        
    elif command == "all":
        print("Running complete pipeline...")
        print("\n" + "="*80)
        print("STEP 1: Data Generation")
        print("="*80)
        from data_generation_hybrid import generate_dataset
        generate_dataset()
        
        print("\n" + "="*80)
        print("STEP 2: Model Training")
        print("="*80)
        from train_model import train_model
        train_model()
        
        print("\n" + "="*80)
        print("STEP 3: Benchmarking")
        print("="*80)
        from benchmark import run_benchmark
        run_benchmark()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        
    else:
        print(f"Unknown command: {command}")
        print_usage()

if __name__ == "__main__":
    main()