"""
Verilog Parser - Compatible with BenchParser Interface

Parses gate-level Verilog and provides the same API as BenchParser.
Supports:
- Basic gates (AND, OR, NOT, NAND, NOR, XOR, XNOR, BUF)
- DFFs (with full-scan assumption)
- Multi-module designs (flattened)
"""

import re

class VerilogParser:
    """
    Parser for gate-level Verilog with BenchParser-compatible API.
    
    For full-scan designs:
    - DFF outputs (Q) are treated as Pseudo Primary Inputs (PPIs)
    - DFF inputs (D) are treated as Pseudo Primary Outputs (PPOs)
    """
    
    # Gate type mapping
    GATE_MAPPINGS = {
        'and': 'AND',
        'or': 'OR',
        'not': 'NOT',
        'nand': 'NAND',
        'nor': 'NOR',
        'xor': 'XOR',
        'xnor': 'XNOR',
        'buf': 'BUFF',
        'buffer': 'BUFF',
        # DFF variants
        'dff': 'DFF',
        'DFF': 'DFF',
    }
    
    def __init__(self, verilog_file):
        self.verilog_file = verilog_file
        
        # Primary Inputs/Outputs
        self.inputs = []
        self.outputs = []
        
        # Pseudo Inputs/Outputs (from DFFs)
        self.ppis = []
        self.ppos = []
        
        # Combined
        self.all_inputs = []
        self.all_outputs = []
        
        # Gate structure
        self.gates = []
        self.gate_dict = {}
        
        # DFF tracking
        self.dffs = []
        self.dff_map = {}
        
        # Back edges
        self.back_edges = {}
        
        # Variable mapping
        self.var_map = {}
        
        # Internal wires
        self.wires = []
        
        # Parse the file
        self._parse()
    
    def _parse(self):
        """Parse Verilog file and populate all data structures."""
        with open(self.verilog_file, 'r') as f:
            content = f.read()
        
        # Remove comments
        content = self._remove_comments(content)
        
        # Extract module(s)
        modules = self._extract_modules(content)
        
        # For now, process first module (can extend to multi-module)
        if not modules:
            raise ValueError("No module found in Verilog file")
        
        module_content = modules[0]
        
        # Parse declarations
        self._parse_ports(module_content)
        self._parse_wires(module_content)
        self._parse_instances(module_content)
        
        # Build combined lists
        self.all_inputs = list(dict.fromkeys(self.inputs + self.ppis))
        self.all_outputs = list(dict.fromkeys(self.outputs + self.ppos))
    
    def _remove_comments(self, content):
        """Remove single-line and multi-line comments."""
        # Remove single-line comments
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        return content
    
    def _extract_modules(self, content):
        """Extract module definitions."""
        pattern = r'module\s+(\w+)\s*\((.*?)\);(.*?)endmodule'
        matches = re.findall(pattern, content, re.DOTALL)
        return [match[2] for match in matches]  # Return module bodies
    
    def _parse_ports(self, content):
        """Parse input and output declarations."""
        # Match: input [width] name1, name2, ...;
        input_pattern = r'input\s+(?:\[.*?\]\s+)?([^;]+);'
        output_pattern = r'output\s+(?:\[.*?\]\s+)?([^;]+);'
        
        for match in re.finditer(input_pattern, content):
            ports = match.group(1).split(',')
            for port in ports:
                name = port.strip()
                name = re.sub(r'\[.*?\]', '', name).strip()  # Remove bus indices
                if name and name not in self.inputs:
                    self.inputs.append(name)
        
        for match in re.finditer(output_pattern, content):
            ports = match.group(1).split(',')
            for port in ports:
                name = port.strip()
                name = re.sub(r'\[.*?\]', '', name).strip()
                if name and name not in self.outputs:
                    self.outputs.append(name)
    
    def _parse_wires(self, content):
        """Parse wire declarations."""
        wire_pattern = r'wire\s+(?:\[.*?\]\s+)?([^;]+);'
        
        for match in re.finditer(wire_pattern, content):
            wires = match.group(1).split(',')
            for wire in wires:
                name = wire.strip()
                name = re.sub(r'\[.*?\]', '', name).strip()
                if name and name not in self.wires:
                    self.wires.append(name)
    
    def _parse_instances(self, content):
        """Parse gate instances."""
        # Pattern: gate_type [#params] inst_name (port_list);
        instance_pattern = r'(\w+)\s+(?:#\(.*?\)\s+)?(\w+)\s*\((.*?)\);'
        
        for match in re.finditer(instance_pattern, content, re.DOTALL):
            gate_type_raw = match.group(1).lower()
            inst_name = match.group(2)
            port_list = match.group(3)
            
            # Skip non-gate instances
            if gate_type_raw not in self.GATE_MAPPINGS:
                continue
            
            gate_type = self.GATE_MAPPINGS[gate_type_raw]
            
            # Parse ports
            ports = self._parse_port_connections(port_list)
            
            if not ports:
                continue
            
            # First port is output, rest are inputs (standard Verilog convention)
            if len(ports) < 1:
                continue
            
            output = ports[0]
            inputs = ports[1:] if len(ports) > 1 else []
            
            # Handle DFFs specially
            if gate_type == 'DFF':
                # DFF output becomes PPI
                self.ppis.append(output)
                
                # DFF input (D) becomes PPO
                if len(inputs) > 0:
                    d_input = inputs[0]
                    self.ppos.append(d_input)
                    
                    # Track DFF relationship
                    self.dffs.append((output, d_input))
                    self.dff_map[output] = d_input
            else:
                # Regular gate
                self.gates.append((output, gate_type, inputs))
                self.gate_dict[output] = (gate_type, inputs)
                
                # Build back edges
                for inp in inputs:
                    if inp not in self.back_edges:
                        self.back_edges[inp] = []
                    self.back_edges[inp].append(output)
    
    def _parse_port_connections(self, port_list):
        """
        Parse port connections from instance.
        Supports both positional and named connections.
        """
        ports = []
        port_list = port_list.strip()
        
        # Check if named connections (.port(wire))
        if '.(' in port_list or '.' in port_list:
            # Named connections: .Q(out), .D(in), .CLK(clk)
            named_pattern = r'\.(\w+)\s*\(\s*(\w+)\s*\)'
            
            connections = {}
            for match in re.finditer(named_pattern, port_list):
                port_name = match.group(1)
                wire_name = match.group(2)
                connections[port_name] = wire_name
            
            # Standard port order: output first, then inputs
            # For gates: typically .Y(out), .A(in1), .B(in2), ...
            # For DFF: .Q(out), .D(in), .CLK(clk)
            
            # Try to extract output (common names: Y, Q, OUT, Z)
            output_candidates = ['Y', 'Q', 'OUT', 'Z', 'O']
            for candidate in output_candidates:
                if candidate in connections:
                    ports.append(connections[candidate])
                    break
            
            # Extract inputs (everything else except CLK, RST, etc)
            skip_ports = ['CLK', 'CLOCK', 'RST', 'RESET', 'SET', 'CLEAR']
            for port_name, wire_name in connections.items():
                if port_name not in output_candidates and port_name not in skip_ports:
                    if wire_name not in ports:
                        ports.append(wire_name)
        else:
            # Positional connections: (out, in1, in2, ...)
            wires = [w.strip() for w in port_list.split(',')]
            ports = [w for w in wires if w]
        
        return ports
    
    # =========================================================================
    # BenchParser-Compatible API
    # =========================================================================
    
    def get_all_wires(self):
        """Get all wire names in the circuit."""
        wires = set(self.all_inputs + self.all_outputs + self.wires)
        for out, _, inputs in self.gates:
            wires.add(out)
            wires.update(inputs)
        return sorted(list(wires))
    
    def build_var_map(self):
        """Build variable mapping for SAT solver (1-indexed)."""
        if self.var_map:
            return self.var_map
        
        next_var = 1
        for wire in self.get_all_wires():
            if wire not in self.var_map:
                self.var_map[wire] = next_var
                next_var += 1
        return self.var_map
    
    def get_fanout(self, wire_name):
        """Get all gates driven by a wire."""
        return self.back_edges.get(wire_name, [])
    
    def get_fanin(self, wire_name):
        """Get the gate driving a wire."""
        if wire_name in self.gate_dict:
            return self.gate_dict[wire_name][1]
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
        return (f"VerilogParser({self.verilog_file})\n"
                f"  PIs: {len(self.inputs)}, POs: {len(self.outputs)}\n"
                f"  PPIs: {len(self.ppis)}, PPOs: {len(self.ppos)}\n"
                f"  Gates: {len(self.gates)}, DFFs: {len(self.dffs)}")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test with a simple Verilog file
    test_verilog = """
    module test_circuit (A, B, C, Y);
        input A, B, C;
        output Y;
        wire n1, n2;
        
        and g1 (n1, A, B);
        or  g2 (n2, n1, C);
        not g3 (Y, n2);
    endmodule
    """
    
    # Write test file
    with open("test_circuit.v", "w") as f:
        f.write(test_verilog)
    
    # Parse
    parser = VerilogParser("test_circuit.v")
    print(parser)
    print("\nGates:", parser.gates)
    print("Inputs:", parser.inputs)
    print("Outputs:", parser.outputs)