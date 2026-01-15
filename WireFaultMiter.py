import os
from BenchParser import BenchParser

class WireFaultMiter:
    def __init__(self, bench_file):
        self.bench_file = bench_file
        
        # Use shared parser
        self.parser = BenchParser(bench_file)
        
        # Extract data from parser
        self.inputs = self.parser.all_inputs      # PIs + PPIs
        self.outputs = self.parser.all_outputs    # POs + PPOs
        self.gates = self.parser.gates
        
        # Build variable map (Deterministic from Parser)
        self.var_map = self.parser.build_var_map()
        self.next_var = len(self.var_map) + 1
        
        # Faulty circuit mapping
        self.faulty_map = {}
        
        self.scan_inputs = self.parser.ppis
        self.scan_outputs = self.parser.ppos

    def _get_var(self, name):
        if name not in self.var_map:
            self.var_map[name] = self.next_var
            self.next_var += 1
        return self.var_map[name]

    def build_miter(self, fault_wire, fault_type=None, force_diff=1):
        clauses = []
        
        # --- 1. Good Circuit ---
        # self.gates is a list (Deterministic order from file)
        for out, g_type, inputs in self.gates:
            self._add_gate_clauses(clauses, self.var_map[out], g_type, [self.var_map[i] for i in inputs])
            
        # --- 2. Faulty Circuit ---
        # Map inputs to same vars, but internal wires get new vars
        # Dict insertion order is preserved in Python 3.7+, but iterating input list is safer
        self.faulty_map = {}
        for name in self.inputs:
            self.faulty_map[name] = self.var_map[name]
            
        for out, _, _ in self.gates:
            if out not in self.faulty_map:
                self.faulty_map[out] = self.next_var
                self.next_var += 1
                
        # Inject Fault (Stuck-at)
        if fault_wire in self.faulty_map:
            fault_gate_var = self.faulty_map[fault_wire]
            if fault_type == 1: clauses.append([fault_gate_var])   # Stuck-at-1
            elif fault_type == 0: clauses.append([-fault_gate_var]) # Stuck-at-0

        for out, g_type, inputs in self.gates:
            # If gate drives the fault wire, disconnect it (fault overrides)
            if out == fault_wire:
                continue

            out_var = self.faulty_map[out]
            in_vars = [self.faulty_map.get(i) for i in inputs]
            if None in in_vars: continue 
            self._add_gate_clauses(clauses, out_var, g_type, in_vars)

        # --- 3. Miter Comparator (XOR Outputs) ---
        miter_out = self.next_var; self.next_var += 1
        diff_vars = []
        
        # --- FIX FOR DETERMINISM ---
        # Old: unique_outputs = list(set(self.outputs)) -> Random order!
        # New: Sorted list -> Fixed order
        unique_outputs = sorted(list(set(self.outputs)))
        
        for out in unique_outputs:
            if out not in self.var_map or out not in self.faulty_map: continue
            
            good = self.var_map[out]
            bad = self.faulty_map[out]
            diff = self.next_var; self.next_var += 1
            
            # XOR Logic: (Good != Bad) -> Diff
            # (-a -b -c), (a b -c), (-a b c), (a -b c)
            clauses.append([-good, -bad, -diff])
            clauses.append([good, bad, -diff])
            clauses.append([-good, bad, diff])
            clauses.append([good, -bad, diff])
            diff_vars.append(diff)
            
        # Big OR Gate (Any difference triggers Miter)
        big_or = [-miter_out]
        for d in diff_vars:
            clauses.append([-d, miter_out])
            big_or.append(d)
        clauses.append(big_or)
        clauses.append([miter_out]) # Force Miter = 1
        
        return clauses

    def _add_gate_clauses(self, clauses, out, g_type, inputs):
        if g_type == 'AND':
            for i in inputs: clauses.append([-out, i])
            clauses.append([out] + [-i for i in inputs])
        elif g_type == 'OR':
            for i in inputs: clauses.append([out, -i])
            clauses.append([-out] + inputs)
        elif g_type == 'NOT':
            clauses.append([-out, -inputs[0]])
            clauses.append([out, inputs[0]])
        elif g_type == 'NAND':
            for i in inputs: clauses.append([out, i])
            clauses.append([-out] + [-i for i in inputs])
        elif g_type == 'NOR':
            for i in inputs: clauses.append([-out, -i])
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