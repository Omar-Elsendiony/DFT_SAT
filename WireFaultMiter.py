import os
import random
from collections import deque
from pysat.solvers import Glucose3, Minisat22
from BenchParser import BenchParser

class WireFaultMiter:
    def __init__(self, bench_file):
        self.bench_file = bench_file
        self.parser = BenchParser(bench_file)
        self.inputs = self.parser.all_inputs      
        self.outputs = self.parser.all_outputs    
        self.gates = self.parser.gates
        self.var_map = self.parser.build_var_map()
        self.next_var = len(self.var_map) + 1
        self.faulty_map = {}
        self.scan_inputs = self.parser.ppis
        self.scan_outputs = self.parser.ppos

    def _get_var(self, name):
        if name not in self.var_map:
            self.var_map[name] = self.next_var
            self.next_var += 1
        return self.var_map[name]

    def build_miter(self, fault_wire, fault_type=None, force_diff=1):
        """
        Build miter circuit for fault detection.
        fault_type: 0 for SA0, 1 for SA1, None for no fault injection
        """
        clauses = []
        
        # --- 1. Good Circuit ---
        for out, g_type, inputs in self.gates:
            self._add_gate_clauses(clauses, self.var_map[out], g_type, [self.var_map[i] for i in inputs])
            
        # --- 2. Faulty Circuit ---
        self.faulty_map = {name: self.var_map[name] for name in self.inputs}
        for out, _, _ in self.gates:
            if out not in self.faulty_map:
                self.faulty_map[out] = self.next_var
                self.next_var += 1
                
        # Inject Fault
        if fault_wire in self.faulty_map and fault_type is not None:
            f_var = self.faulty_map[fault_wire]
            if fault_type == 1: 
                clauses.append([f_var])   # SA1
            elif fault_type == 0: 
                clauses.append([-f_var])  # SA0

        for out, g_type, inputs in self.gates:
            if out == fault_wire: 
                continue
            out_var = self.faulty_map[out]
            in_vars = [self.faulty_map.get(i) for i in inputs]
            if None in in_vars: 
                continue 
            self._add_gate_clauses(clauses, out_var, g_type, in_vars)

        # --- 3. Miter Comparator ---
        miter_out = self.next_var
        self.next_var += 1
        diff_vars = []
        unique_outputs = sorted(list(set(self.outputs)))
        
        for out in unique_outputs:
            if out not in self.var_map or out not in self.faulty_map: 
                continue
            good = self.var_map[out]
            bad = self.faulty_map[out]
            diff = self.next_var
            self.next_var += 1
            
            # XOR logic
            clauses.extend([
                [-good, -bad, -diff], 
                [good, bad, -diff], 
                [-good, bad, diff], 
                [good, -bad, diff]
            ])
            diff_vars.append(diff)
            
        # OR gate
        big_or = [-miter_out]
        for d in diff_vars:
            clauses.append([-d, miter_out])
            big_or.append(d)
        clauses.append(big_or)
        clauses.append([miter_out])  # Force miter output = 1
        
        return clauses

    def _add_gate_clauses(self, clauses, out, g_type, inputs):
        if g_type == 'AND':
            for i in inputs: 
                clauses.append([-out, i])
            clauses.append([out] + [-i for i in inputs])
        elif g_type == 'OR':
            for i in inputs: 
                clauses.append([out, -i])
            clauses.append([-out] + inputs)
        elif g_type == 'NOT':
            clauses.append([-out, -inputs[0]])
            clauses.append([out, inputs[0]])
        elif g_type == 'NAND':
            for i in inputs: 
                clauses.append([out, i])
            clauses.append([-out] + [-i for i in inputs])
        elif g_type == 'NOR':
            for i in inputs: 
                clauses.append([-out, -i])
            clauses.append([out] + inputs)
        elif g_type == 'XOR':
            if len(inputs) == 2:
                a, b = inputs
                clauses.append([-out, -a, -b])
                clauses.append([-out, a, b])
                clauses.append([out, -a, b])
                clauses.append([out, a, -b])
        elif g_type == 'BUFF':
             clauses.append([-out, inputs[0]])
             clauses.append([out, -inputs[0]])