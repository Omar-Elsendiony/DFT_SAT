"""
Python wrapper for Glucose C++ SAT Solver

This module provides a Python interface to the Glucose SAT solver written in C++.
It handles:
1. Compilation of Glucose (if needed)
2. DIMACS file generation
3. Subprocess communication with the Glucose binary
4. Result parsing and analysis
"""

import subprocess
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import re
from pysat.formula import CNF


class GlucoseSolverWrapper:
    """
    Wrapper for Glucose C++ SAT Solver.
    
    Features:
    - Automatic compilation detection
    - DIMACS format support
    - Resource tracking (conflicts, decisions, propagations)
    - Variable ordering hints
    """
    
    def __init__(self, glucose_dir: Optional[str] = None):
        """
        Initialize Glucose solver wrapper.
        
        Args:
            glucose_dir: Path to glucose directory (default: ./glucose)
        """
        self.glucose_dir = glucose_dir or os.path.join(
            os.path.dirname(__file__), "glucose"
        )
        self.glucose_binary = None
        self._find_or_build_glucose()
    
    def _find_or_build_glucose(self):
        """Find Glucose binary or build if not present."""
        
        # Check parallel solver first
        candidates = [
            os.path.join(self.glucose_dir, "parallel", "glucose"),
            os.path.join(self.glucose_dir, "simp", "glucose"),
            os.path.join(self.glucose_dir, "core", "glucose"),
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                self.glucose_binary = candidate
                print(f"✓ Found Glucose binary: {candidate}")
                return
        
        # If not found, offer to build
        print(f"⚠ Glucose binary not found in {self.glucose_dir}")
        print(f"  Run: cd {self.glucose_dir}/parallel && make")
        raise FileNotFoundError(
            f"Glucose solver not found. Please build it first."
        )
    
    def solve_dimacs(self, dimacs_file: str, timeout: int = 300, 
                     verbose: bool = True) -> Dict:
        """
        Solve a SAT problem in DIMACS format.
        
        Args:
            dimacs_file: Path to DIMACS CNF file
            timeout: Time limit in seconds
            verbose: Print solver output
            
        Returns:
            Dictionary with results (satisfiable, conflicts, decisions, etc.)
        """
        if not os.path.exists(dimacs_file):
            raise FileNotFoundError(f"DIMACS file not found: {dimacs_file}")
        
        print(f"\n{'='*70}")
        print(f"Glucose C++ SAT Solver")
        print(f"Input: {os.path.basename(dimacs_file)}")
        print(f"Timeout: {timeout}s")
        print(f"{'='*70}\n")
        
        try:
            result = subprocess.run(
                [self.glucose_binary, dimacs_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = result.stdout + result.stderr
            
            if verbose:
                print(output)
            
            # Parse results
            return self._parse_glucose_output(output, result.returncode)
            
        except subprocess.TimeoutExpired:
            print(f"✗ Solver timed out after {timeout}s")
            return {
                'satisfiable': None,
                'timeout': True,
                'error': 'Timeout exceeded'
            }
        except Exception as e:
            print(f"✗ Solver error: {e}")
            return {
                'satisfiable': None,
                'error': str(e)
            }
    
    def solve_from_cnf(self, cnf: CNF, timeout: int = 300, 
                       verbose: bool = True) -> Dict:
        """
        Solve a CNF formula directly.
        
        Args:
            cnf: PySAT CNF object
            timeout: Time limit in seconds
            verbose: Print solver output
            
        Returns:
            Dictionary with results
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
            dimacs_file = f.name
            cnf.to_file(dimacs_file)
        
        try:
            return self.solve_dimacs(dimacs_file, timeout, verbose)
        finally:
            if os.path.exists(dimacs_file):
                os.remove(dimacs_file)
    
    def _parse_glucose_output(self, output: str, return_code: int) -> Dict:
        """
        Parse Glucose solver output.
        
        Args:
            output: Combined stdout/stderr from solver
            return_code: Process return code
            
        Returns:
            Parsed results dictionary
        """
        results = {
            'return_code': return_code,
            'raw_output': output,
            'satisfiable': None,
            'conflicts': None,
            'decisions': None,
            'propagations': None,
            'variables': None,
            'clauses': None,
            'cpu_time': None
        }
        
        # Parse main result (SATISFIABLE or UNSATISFIABLE)
        if 'SATISFIABLE' in output:
            results['satisfiable'] = True
        elif 'UNSATISFIABLE' in output:
            results['satisfiable'] = False
        elif return_code == 10:
            results['satisfiable'] = True
        elif return_code == 20:
            results['satisfiable'] = False
        
        # Parse statistics
        patterns = {
            'conflicts': r'conflicts\s*:\s*(\d+)',
            'decisions': r'decisions\s*:\s*(\d+)',
            'propagations': r'propagations\s*:\s*(\d+)',
            'variables': r'variables\s*:\s*(\d+)',
            'clauses': r'clauses\s*:\s*(\d+)',
            'cpu_time': r'CPU time\s*:\s*([\d.]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                results[key] = float(match.group(1))
        
        # Extract satisfying assignment if SAT
        if results['satisfiable']:
            assignment = self._parse_assignment(output)
            results['model'] = assignment
        
        return results
    
    def _parse_assignment(self, output: str) -> Optional[List[int]]:
        """
        Extract variable assignment from solver output.
        
        Returns:
            List of variable assignments (or None if not found)
        """
        # Look for line starting with 'v' containing the model
        for line in output.split('\n'):
            if line.startswith('v'):
                # Extract integers from the line
                tokens = line[1:].strip().split()
                try:
                    assignment = [int(t) for t in tokens if t != '0']
                    return assignment
                except ValueError:
                    continue
        return None


def demo_glucose_wrapper():
    """Demo: Using Glucose wrapper with a simple CNF."""
    
    # Create simple CNF: (x1 OR x2 OR ~x3) AND (~x1 OR x2)
    cnf = CNF()
    cnf.append([1, 2, -3])
    cnf.append([-1, 2])
    
    # Solve
    solver = GlucoseSolverWrapper()
    results = solver.solve_from_cnf(cnf, timeout=10)
    
    print("\nResults:")
    print(f"Satisfiable: {results['satisfiable']}")
    print(f"Conflicts: {results['conflicts']}")
    print(f"Decisions: {results['decisions']}")
    if results['model']:
        print(f"Model: {results['model']}")


if __name__ == "__main__":
    demo_glucose_wrapper()
