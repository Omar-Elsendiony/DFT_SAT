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

    # --- OPTIMIZATION METHODS ---

    def get_reachable_outputs(self, fault_wire):
        reachable_pos = set()
        queue = deque([fault_wire])
        visited = {fault_wire}
        while queue:
            wire = queue.popleft()
            if wire in self.outputs: reachable_pos.add(wire)
            if wire in self.parser.back_edges:
                for gate_out in self.parser.back_edges[wire]:
                    if gate_out not in visited:
                        visited.add(gate_out)
                        queue.append(gate_out)
        return list(reachable_pos)

    def get_logic_cone(self, target_outputs, fault_wire):
        relevant_gates = set()
        queue = deque(list(target_outputs) + [fault_wire])
        visited = set(list(target_outputs) + [fault_wire])
        while queue:
            wire = queue.popleft()
            if wire in self.parser.gate_dict:
                g_type, inputs = self.parser.gate_dict[wire]
                relevant_gates.add((wire, g_type, tuple(inputs)))
                for inp in inputs:
                    if inp not in visited:
                        visited.add(inp)
                        queue.append(inp)
        return sorted(list(relevant_gates), key=lambda x: x[0])

    def get_branchless_implications(self, fault_wire, target_val):
        forced = {}
        curr, val = fault_wire, target_val
        while True:
            if curr in self.inputs:
                forced[curr] = val
                break
            if curr not in self.parser.gate_dict: break
            
            g_type, inputs = self.parser.gate_dict[curr]
            
            if g_type == 'NOT': n_val = 1 - val
            elif g_type == 'BUFF': n_val = val
            elif g_type == 'AND' and val == 1: 
                for i in inputs: forced[i] = 1; 
                break
            elif g_type == 'OR' and val == 0:
                for i in inputs: forced[i] = 0
                break
            elif g_type == 'NAND' and val == 0:
                for i in inputs: forced[i] = 1
                break
            elif g_type == 'NOR' and val == 1:
                for i in inputs: forced[i] = 0
                break
            else: break
            
            forced[inputs[0]] = n_val
            curr, val = inputs[0], n_val
        return forced

    def solve_fault_specific_cones(self, fault_wire, fault_type=1, gnn_hints=None):
        """
        Solves using Split-Cone + Heuristic Sorting + Branchless + GNN Hints.
        """
        possible_outputs = self.get_reachable_outputs(fault_wire)
        if not possible_outputs: return None, 0
        
        # Sample outputs to avoid hang on large circuits
        sample_size = min(20, len(possible_outputs))
        sampled_subset = random.sample(possible_outputs, sample_size)
        remaining_subset = list(set(possible_outputs) - set(sampled_subset))
        
        candidates = []
        for po in sampled_subset:
            cone = self.get_logic_cone([po], fault_wire)
            candidates.append((len(cone), po))
        
        candidates.sort(key=lambda x: x[0])
        sorted_outputs = [x[1] for x in candidates] + remaining_subset

        orig_gates, orig_outputs = self.gates, self.outputs
        
        activation_val = 1 if fault_type == 0 else 0
        forced_map = self.get_branchless_implications(fault_wire, activation_val)
        
        final_assignment = None
        total_conflicts = 0
        
        for target_po in sorted_outputs:
            cone_gates = self.get_logic_cone([target_po], fault_wire)
            self.gates, self.outputs = cone_gates, [target_po]
            
            # 1. Reset Vars (Inputs must keep original IDs for hints)
            self.var_map = {name: i+1 for i, name in enumerate(self.parser.all_inputs)}
            self.next_var = len(self.var_map) + 1
            
            # 2. POPULATE VAR_MAP FOR INTERNAL WIRES (The Missing Fix) 
            for out, _, inputs in self.gates:
                if out not in self.var_map:
                    self.var_map[out] = self.next_var
                    self.next_var += 1
                for inp in inputs:
                    if inp not in self.var_map:
                        self.var_map[inp] = self.next_var
                        self.next_var += 1
            
            assumptions = []
            for name, val in forced_map.items():
                if name in self.var_map:
                    lit = self.var_map[name]
                    assumptions.append(lit if val == 1 else -lit)

            phases = []
            if gnn_hints:
                for name, prob in gnn_hints.items():
                    if name in self.var_map:
                        lit = self.var_map[name]
                        phases.append(lit if prob > 0.5 else -lit)

            clauses = self.build_miter(fault_wire, fault_type)
            with Minisat22(bootstrap_with=clauses) as solver:
                if phases: solver.set_phases(phases)
                
                # Timeout for bad cones
                solver.conf_budget(2000) 
                
                result = solver.solve(assumptions=assumptions)
                total_conflicts += solver.accum_stats()['conflicts']
                
                if result:
                    final_assignment = self._extract_assignment(solver.get_model())
                    break 
        
        self.gates, self.outputs = orig_gates, orig_outputs
        return final_assignment, total_conflicts

    def _extract_assignment(self, model):
        assign = {}
        model_set = set(model)
        for inp in self.inputs:
            if inp in self.var_map:
                vid = self.var_map[inp]
                if vid in model_set: assign[inp] = 1
                elif -vid in model_set: assign[inp] = 0
                else: assign[inp] = 'X'
        return assign

    # --- STANDARD BUILDER ---
    def build_miter(self, fault_wire, fault_type=None, force_diff=1):
        clauses = []
        for out, g_type, inputs in self.gates:
            self._add_gate_clauses(clauses, self.var_map[out], g_type, [self.var_map[i] for i in inputs])
            
        self.faulty_map = {name: self.var_map[name] for name in self.inputs}
        for out, _, _ in self.gates:
            if out not in self.faulty_map:
                self.faulty_map[out] = self.next_var
                self.next_var += 1
                
        if fault_wire in self.faulty_map:
            f_var = self.faulty_map[fault_wire]
            if fault_type == 1: clauses.append([f_var])
            elif fault_type == 0: clauses.append([-f_var])

        for out, g_type, inputs in self.gates:
            if out == fault_wire: continue
            out_var = self.faulty_map[out]
            in_vars = [self.faulty_map.get(i) for i in inputs]
            if None in in_vars: continue 
            self._add_gate_clauses(clauses, out_var, g_type, in_vars)

        miter_out = self.next_var; self.next_var += 1
        diff_vars = []
        unique_outputs = sorted(list(set(self.outputs)))
        
        for out in unique_outputs:
            if out not in self.var_map or out not in self.faulty_map: continue
            good = self.var_map[out]
            bad = self.faulty_map[out]
            diff = self.next_var; self.next_var += 1
            clauses.extend([[-good, -bad, -diff], [good, bad, -diff], [-good, bad, diff], [good, -bad, diff]])
            diff_vars.append(diff)
            
        big_or = [-miter_out]
        for d in diff_vars:
            clauses.append([-d, miter_out])
            big_or.append(d)
        clauses.append(big_or)
        clauses.append([miter_out]) 
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