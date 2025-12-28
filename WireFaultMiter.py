import os

class WireFaultMiter:
    def __init__(self, bench_file):
        self.bench_file = bench_file
        self.inputs = []
        self.outputs = []
        self.gates = []
        self.var_map = {}
        self.faulty_map = {}
        self.next_var = 1
        
        # Track Scan Chains for debug/miter construction
        self.scan_inputs = []
        self.scan_outputs = []

    def _get_var(self, name):
        if name not in self.var_map:
            self.var_map[name] = self.next_var
            self.next_var += 1
        return self.var_map[name]

    def _parse_bench(self):
        with open(self.bench_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                
                if line.startswith('INPUT'):
                    name = line[line.find('(')+1:line.find(')')]
                    self.inputs.append(name)
                    self._get_var(name)
                    
                elif line.startswith('OUTPUT'):
                    name = line[line.find('(')+1:line.find(')')]
                    self.outputs.append(name)
                    self._get_var(name)
                    
                elif '=' in line:
                    parts = line.split('=')
                    out = parts[0].strip()
                    rhs = parts[1].strip()
                    g_type = rhs[:rhs.find('(')].strip().upper()
                    in_str = rhs[rhs.find('(')+1:-1]
                    inputs = [x.strip() for x in in_str.split(',')] if in_str else []
                    
                    # --- FIX: Handle Flip-Flops (Full-Scan) ---
                    if g_type == 'DFF':
                        # Output of DFF -> Pseudo Input
                        self.inputs.append(out)
                        self.scan_inputs.append(out)
                        self._get_var(out)
                        
                        # Input to DFF -> Pseudo Output
                        if len(inputs) > 0:
                            self.outputs.append(inputs[0])
                            self.scan_outputs.append(inputs[0])
                            self._get_var(inputs[0])
                        # Do NOT add to self.gates (loop cut)
                    else:
                        self.gates.append((out, g_type, inputs))
                        self._get_var(out)
                        for i in inputs: self._get_var(i)

    def build_miter(self, fault_wire, fault_type=None, force_diff=1):
        clauses = []
        
        # --- 1. Good Circuit ---
        for out, g_type, inputs in self.gates:
            self._add_gate_clauses(clauses, self.var_map[out], g_type, [self.var_map[i] for i in inputs])
            
        # --- 2. Faulty Circuit ---
        # Map inputs to same vars, but internal wires get new vars
        self.faulty_map = {name: self.var_map[name] for name in self.inputs}
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
            # --- CRITICAL FIX: Disconnect Logic for Faulty Wire ---
            # If this gate drives the wire that is currently stuck, 
            # we must NOT generate clauses for it. The wire is controlled by the fault, not the gate.
            if out == fault_wire:
                continue
            # -----------------------------------------------------

            out_var = self.faulty_map[out]
            in_vars = [self.faulty_map.get(i) for i in inputs]
            if None in in_vars: continue # Skip if inputs are missing (rare scan edge case)
            self._add_gate_clauses(clauses, out_var, g_type, in_vars)

        # --- 3. Miter Comparator (XOR Outputs) ---
        miter_out = self.next_var; self.next_var += 1
        diff_vars = []
        
        unique_outputs = list(set(self.outputs)) # Handles POs and PPOs
        
        for out in unique_outputs:
            if out not in self.var_map or out not in self.faulty_map: continue
            
            good = self.var_map[out]
            bad = self.faulty_map[out]
            diff = self.next_var; self.next_var += 1
            
            # XOR Logic
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