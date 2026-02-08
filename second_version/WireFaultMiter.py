"""
WireFaultMiter with Complete ATPG Cone Extraction

Includes proper handling of:
1. Fan-in (activation)
2. Fan-out (propagation path)  
3. Side inputs (other logic affecting propagation)
"""

import os
import random
from collections import deque
from pysat.solvers import Glucose3, Minisat22

try:
    from UnifiedParser import UnifiedParser as Parser
except ImportError:
    from BenchParser import BenchParser as Parser

class WireFaultMiter:
    def __init__(self, circuit_file):
        """Initialize miter for fault detection."""
        self.circuit_file = circuit_file
        self.parser = Parser(circuit_file)
        
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

    def get_reachable_outputs(self, gate_name):
        """Get all primary outputs reachable from a gate."""
        reachable = set()
        visited = set()
        queue = deque([gate_name])
        
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            
            if node in self.outputs:
                reachable.add(node)
            
            fanout = self.parser.get_fanout(node)
            for next_gate in fanout:
                if next_gate not in visited:
                    queue.append(next_gate)
        
        return list(reachable)

    def get_complete_atpg_cone(self, gate_name, target_output):
        """
        Extract the COMPLETE ATPG cone including side inputs.
        
        This includes:
        1. Activation cone: inputs → fault (fan-in)
        2. Propagation cone: fault → output (fan-out path)
        3. Side input cones: other logic affecting propagation gates ⭐
        
        Args:
            gate_name: Fault site
            target_output: Target primary output (used to limit scope)
            
        Returns:
            Complete ATPG cone as list of (output, gate_type, inputs) tuples
        """
        # Step 1: Trace the propagation path (fault → target output)
        propagation_path = set()
        visited_prop = set()
        
        def trace_propagation(node, target):
            """Find all gates on path from fault to specific output"""
            if node in visited_prop:
                return
            visited_prop.add(node)
            
            # Stop if we reached the target
            if node == target:
                return
            
            # Add to path if it's a gate (not the fault itself yet)
            if node != gate_name and node in self.parser.gate_dict:
                propagation_path.add(node)
            
            # Continue tracing only if we haven't reached target
            if node != target:
                fanout = self.parser.get_fanout(node)
                for next_gate in fanout:
                    if next_gate not in visited_prop:
                        trace_propagation(next_gate, target)
        
        trace_propagation(gate_name, target_output)
        
        # Step 2: For EACH propagation gate, get its COMPLETE fan-in
        # This captures side inputs! ⭐
        side_logic = set()
        propagation_cone = []
        
        for prop_gate in propagation_path:
            # Add the propagation gate itself
            if prop_gate in self.parser.gate_dict:
                g_type, inputs = self.parser.gate_dict[prop_gate]
                propagation_cone.append((prop_gate, g_type, inputs))
                
                # Get ALL logic feeding this propagation gate
                # (excluding the fault gate to avoid double-counting)
                gate_fanin = self._get_fanin_recursive(prop_gate, stop_at=gate_name)
                for fanin_gate, _, _ in gate_fanin:
                    side_logic.add(fanin_gate)
        
        # Step 3: Get fault activation cone (fan-in to fault)
        activation_cone = self._get_fanin_recursive(gate_name, stop_at=None)
        
        # Step 4: Build side input cone (gates in side_logic but not activation)
        side_cone = []
        activation_gates = set([g[0] for g in activation_cone])
        
        for gate in side_logic:
            if gate not in activation_gates and gate in self.parser.gate_dict:
                g_type, inputs = self.parser.gate_dict[gate]
                side_cone.append((gate, g_type, inputs))
        
        # Step 5: Add the fault gate itself
        fault_gate = []
        if gate_name in self.parser.gate_dict:
            g_type, inputs = self.parser.gate_dict[gate_name]
            fault_gate = [(gate_name, g_type, inputs)]
        
        # Step 6: Combine all components and deduplicate
        all_gates = activation_cone + fault_gate + side_cone + propagation_cone
        seen = set()
        complete_cone = []
        
        for gate in all_gates:
            if gate[0] not in seen:
                seen.add(gate[0])
                complete_cone.append(gate)
        
        return complete_cone

    def _get_fanin_recursive(self, gate_name, stop_at=None):
        """
        Helper to get fan-in cone with optional stop gate.
        
        Args:
            gate_name: Gate to analyze
            stop_at: Gate to stop at (e.g., the fault gate), None to traverse fully
            
        Returns:
            List of (output, gate_type, inputs) tuples in fan-in cone
        """
        cone = []
        visited = set()
        
        def dfs_backward(node):
            if node in visited or node in self.inputs:
                return
            if stop_at is not None and node == stop_at:
                return
            visited.add(node)
            
            if node in self.parser.gate_dict:
                g_type, inputs = self.parser.gate_dict[node]
                cone.append((node, g_type, inputs))
                for inp in inputs:
                    dfs_backward(inp)
        
        dfs_backward(gate_name)
        return cone

    def get_cone_inputs(self, cone_gates):
        """Extract all primary inputs referenced by a cone."""
        cone_inputs = set()
        for _, _, inputs in cone_gates:
            for inp in inputs:
                if inp in self.inputs:
                    cone_inputs.add(inp)
        return cone_inputs

    def build_miter(self, fault_wire, fault_type=None, force_diff=1):
        """Build miter circuit for fault detection."""
        clauses = []
        
        # 1. Good Circuit
        for out, g_type, inputs in self.gates:
            self._add_gate_clauses(clauses, self.var_map[out], g_type, 
                                  [self.var_map[i] for i in inputs])
            
        # 2. Faulty Circuit
        self.faulty_map = {name: self.var_map[name] for name in self.inputs}
        for out, _, _ in self.gates:
            if out not in self.faulty_map:
                self.faulty_map[out] = self.next_var
                self.next_var += 1
                
        # Inject Fault
        if fault_wire in self.faulty_map and fault_type is not None:
            f_var = self.faulty_map[fault_wire]
            if fault_type == 1: 
                clauses.append([f_var])
            elif fault_type == 0: 
                clauses.append([-f_var])

        for out, g_type, inputs in self.gates:
            if out == fault_wire: 
                continue
            out_var = self.faulty_map[out]
            in_vars = [self.faulty_map.get(i) for i in inputs]
            if None in in_vars: 
                continue 
            self._add_gate_clauses(clauses, out_var, g_type, in_vars)

        # 3. Miter Comparator
        miter_out = self.next_var
        self.next_var += 1
        diff_vars = []
        
        for out in sorted(list(set(self.outputs))):
            if out not in self.var_map or out not in self.faulty_map: 
                continue
            good = self.var_map[out]
            bad = self.faulty_map[out]
            diff = self.next_var
            self.next_var += 1
            
            clauses.extend([
                [-good, -bad, -diff], 
                [good, bad, -diff], 
                [-good, bad, diff], 
                [good, -bad, diff]
            ])
            diff_vars.append(diff)
            
        big_or = [-miter_out]
        for d in diff_vars:
            clauses.append([-d, miter_out])
            big_or.append(d)
        clauses.append(big_or)
        clauses.append([miter_out])
        
        return clauses

    def _add_gate_clauses(self, clauses, out, g_type, inputs):
        """Add CNF clauses for a gate."""
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
            else:
                prev = inputs[0]
                for inp in inputs[1:-1]:
                    temp = self.next_var
                    self.next_var += 1
                    clauses.extend([
                        [-temp, -prev, -inp],
                        [-temp, prev, inp],
                        [temp, -prev, inp],
                        [temp, prev, -inp]
                    ])
                    prev = temp
                a, b = prev, inputs[-1]
                clauses.extend([
                    [-out, -a, -b],
                    [-out, a, b],
                    [out, -a, b],
                    [out, a, -b]
                ])
        elif g_type == 'XNOR':
            if len(inputs) == 2:
                a, b = inputs
                clauses.append([out, -a, -b])
                clauses.append([out, a, b])
                clauses.append([-out, -a, b])
                clauses.append([-out, a, -b])
        elif g_type == 'BUFF':
            clauses.append([-out, inputs[0]])
            clauses.append([out, -inputs[0]])
        else:
            if len(inputs) > 0:
                clauses.append([-out, inputs[0]])
                clauses.append([out, -inputs[0]])