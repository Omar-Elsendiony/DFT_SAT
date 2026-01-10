"""
Shared Bench File Parser for DFT Analysis

This module provides a unified parser for .bench files that:
- Handles full-scan DFFs (Q output as PPI, D input as PPO)
- Tracks back edges for bidirectional graph traversal
- Provides a common data structure for both SAT and GNN models
"""

class BenchParser:
    """
    Unified parser for .bench format files with full-scan DFF support.
    
    For full-scan designs:
    - DFF outputs (Q) are treated as Pseudo Primary Inputs (PPIs)
    - DFF inputs (D) are treated as Pseudo Primary Outputs (PPOs)
    - The circuit is "broken" at flip-flops to eliminate cycles
    """
    
    def __init__(self, bench_file):
        self.bench_file = bench_file
        
        # Primary Inputs/Outputs
        self.inputs = []           # Primary Inputs (PIs)
        self.outputs = []          # Primary Outputs (POs)
        
        # Pseudo Inputs/Outputs (from DFFs)
        self.ppis = []             # Pseudo Primary Inputs (DFF Q outputs)
        self.ppos = []             # Pseudo Primary Outputs (DFF D inputs)
        
        # All inputs/outputs combined
        self.all_inputs = []       # PIs + PPIs
        self.all_outputs = []      # POs + PPOs
        
        # Gate structure
        self.gates = []            # List of (output, gate_type, inputs)
        self.gate_dict = {}        # Map: output_name -> (gate_type, inputs)
        
        # DFF tracking
        self.dffs = []             # List of (Q_output, D_input) tuples
        self.dff_map = {}          # Map: Q_output -> D_input
        
        # Back edges (for reverse traversal)
        self.back_edges = {}       # Map: input_wire -> [gates_it_drives]
        
        # Variable mapping (for SAT solver)
        self.var_map = {}          # Map: wire_name -> variable_id
        
        # Parse the file
        self._parse()
    
    def _parse(self):
        """Parse the bench file and populate all data structures."""
        with open(self.bench_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse INPUT declarations
                if line.startswith('INPUT'):
                    name = line[line.find('(')+1:line.find(')')]
                    self.inputs.append(name)
                    self.all_inputs.append(name)
                    
                # Parse OUTPUT declarations
                elif line.startswith('OUTPUT'):
                    name = line[line.find('(')+1:line.find(')')]
                    self.outputs.append(name)
                    self.all_outputs.append(name)
                    
                # Parse gate definitions
                elif '=' in line:
                    parts = line.split('=')
                    out = parts[0].strip()
                    rhs = parts[1].strip()
                    
                    # Extract gate type and inputs
                    g_type = rhs[:rhs.find('(')].strip().upper()
                    in_str = rhs[rhs.find('(')+1:-1]
                    inputs = [x.strip() for x in in_str.split(',')] if in_str else []
                    
                    # Handle DFFs specially (Full-Scan assumption)
                    if g_type == 'DFF':
                        # Q output (out) becomes a PPI
                        self.ppis.append(out)
                        self.all_inputs.append(out)
                        
                        # D input becomes a PPO
                        if len(inputs) > 0:
                            d_input = inputs[0]
                            self.ppos.append(d_input)
                            self.all_outputs.append(d_input)
                            
                            # Track the DFF relationship
                            self.dffs.append((out, d_input))
                            self.dff_map[out] = d_input
                        
                        # Note: DFFs are NOT added to self.gates
                        # This "breaks" the circuit at flip-flops
                    else:
                        # Regular combinational gate
                        self.gates.append((out, g_type, inputs))
                        self.gate_dict[out] = (g_type, inputs)
                        
                        # Build back edges for reverse traversal
                        for inp in inputs:
                            if inp not in self.back_edges:
                                self.back_edges[inp] = []
                            self.back_edges[inp].append(out)
        
        # Remove duplicates while preserving order
        self.all_inputs = list(dict.fromkeys(self.all_inputs))
        self.all_outputs = list(dict.fromkeys(self.all_outputs))
    
    def get_all_wires(self):
        """Get all wire names in the circuit (inputs, outputs, and internal)."""
        wires = set(self.all_inputs + self.all_outputs)
        for out, _, inputs in self.gates:
            wires.add(out)
            wires.update(inputs)
        return sorted(list(wires))
    
    def build_var_map(self):
        """Build variable mapping for SAT solver (1-indexed)."""
        if self.var_map:
            return self.var_map  # Already built
        
        next_var = 1
        for wire in self.get_all_wires():
            if wire not in self.var_map:
                self.var_map[wire] = next_var
                next_var += 1
        return self.var_map
    
    def get_fanout(self, wire_name):
        """Get all gates driven by a wire (forward edges)."""
        return self.back_edges.get(wire_name, [])
    
    def get_fanin(self, wire_name):
        """Get the gate driving a wire (backward edge)."""
        if wire_name in self.gate_dict:
            return self.gate_dict[wire_name][1]  # Return inputs
        return []
    
    def is_pi(self, wire_name):
        """Check if wire is a Primary Input."""
        return wire_name in self.inputs
    
    def is_po(self, wire_name):
        """Check if wire is a Primary Output."""
        return wire_name in self.outputs
    
    def is_ppi(self, wire_name):
        """Check if wire is a Pseudo Primary Input (DFF Q)."""
        return wire_name in self.ppis
    
    def is_ppo(self, wire_name):
        """Check if wire is a Pseudo Primary Output (DFF D)."""
        return wire_name in self.ppos
    
    def is_dff_output(self, wire_name):
        """Check if wire is a DFF Q output."""
        return wire_name in self.dff_map
    
    def get_dff_input(self, q_output):
        """Get the D input for a DFF Q output."""
        return self.dff_map.get(q_output)
    
    def get_gate_type(self, wire_name):
        """Get the gate type that produces this wire."""
        if wire_name in self.gate_dict:
            return self.gate_dict[wire_name][0]
        elif self.is_ppi(wire_name):
            return 'PPI'
        elif self.is_pi(wire_name):
            return 'INPUT'
        return None
    
    def __repr__(self):
        return (f"BenchParser({self.bench_file})\n"
                f"  PIs: {len(self.inputs)}, POs: {len(self.outputs)}\n"
                f"  PPIs: {len(self.ppis)}, PPOs: {len(self.ppos)}\n"
                f"  Gates: {len(self.gates)}, DFFs: {len(self.dffs)}")
