r"""
VerilogParser - Enhanced for Yosys Output

Handles:
- Yosys header comments
- Escaped identifiers (\a[0], \b[1], etc.)
- Yosys internal gates (\$_AND_, \$_OR_, etc.)
- Both named and positional port connections
"""

import re

class VerilogParser:
    """Parser for gate-level Verilog with BenchParser-compatible API."""
    
    GATE_MAPPINGS = {
        # Standard gates (lowercase)
        'and': 'AND', 'or': 'OR', 'not': 'NOT', 'nand': 'NAND',
        'nor': 'NOR', 'xor': 'XOR', 'xnor': 'XNOR', 'buf': 'BUFF',
        'buffer': 'BUFF', 'dff': 'DFF', 'DFF': 'DFF',
        
        # Yosys internal gates (with $_..._  format)
        '$_and_': 'AND', '$_or_': 'OR', '$_not_': 'NOT', '$_nand_': 'NAND',
        '$_nor_': 'NOR', '$_xor_': 'XOR', '$_xnor_': 'XNOR', '$_buf_': 'BUFF',
        '$_dff_': 'DFF', '$_dffe_': 'DFF',
        '$_mux_': 'MUX', '$_nmux_': 'NMUX',
        '$_aoi3_': 'AOI3', '$_oai3_': 'OAI3',
        '$_aoi4_': 'AOI4', '$_oai4_': 'OAI4',
    }
    
    def __init__(self, verilog_file):
        self.verilog_file = verilog_file
        
        # Data structures
        self.inputs = []
        self.outputs = []
        self.ppis = []
        self.ppos = []
        self.all_inputs = []
        self.all_outputs = []
        self.gates = []
        self.gate_dict = {}
        self.dffs = []
        self.dff_map = {}
        self.back_edges = {}
        self.var_map = {}
        self.wires = []
        
        self._parse()
    
    def _remove_comments(self, content):
        """Remove ALL comments including Yosys headers."""
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # Remove single-line comments
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        return content
    
    def _normalize_identifier(self, name):
        """
        Normalize Verilog identifiers.
        Removes escaped identifier backslash and trailing whitespace.
        
        Examples:
            \\a[0]  -> a[0]
            \\b[1]  -> b[1]
            normal  -> normal
        """
        name = name.strip()
        # Remove leading backslash for escaped identifiers
        if name.startswith('\\'):
            name = name[1:].strip()
        return name
    
    def _parse(self):
        """Parse Verilog file."""
        with open(self.verilog_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Remove comments FIRST
        content = self._remove_comments(content)
        
        # Extract modules
        modules = self._extract_modules(content)
        
        if not modules:
            raise ValueError("No module found in Verilog file")
        
        # Process first module
        module_content = modules[0]
        
        self._parse_ports(module_content)
        self._parse_wires(module_content)
        self._parse_instances(module_content)
        
        # Build combined lists
        self.all_inputs = list(dict.fromkeys(self.inputs + self.ppis))
        self.all_outputs = list(dict.fromkeys(self.outputs + self.ppos))
    
    def _extract_modules(self, content):
        """Extract module definitions."""
        # Handle both escaped and normal identifiers in module name and ports
        pattern = r'module\s+(\S+)\s*\((.*?)\);(.*?)endmodule'
        matches = re.findall(pattern, content, re.DOTALL)
        return [match[2] for match in matches]  # Return module bodies
    
    def _parse_ports(self, content):
        """Parse input/output declarations."""
        # Updated patterns to handle escaped identifiers
        input_pattern = r'input\s+(?:\[.*?\]\s+)?([^;]+);'
        output_pattern = r'output\s+(?:\[.*?\]\s+)?([^;]+);'
        
        for match in re.finditer(input_pattern, content):
            ports = match.group(1).split(',')
            for port in ports:
                name = self._normalize_identifier(port)
                name = re.sub(r'\[.*?\]', '', name).strip()
                if name and name not in self.inputs:
                    self.inputs.append(name)
        
        for match in re.finditer(output_pattern, content):
            ports = match.group(1).split(',')
            for port in ports:
                name = self._normalize_identifier(port)
                name = re.sub(r'\[.*?\]', '', name).strip()
                if name and name not in self.outputs:
                    self.outputs.append(name)
    
    def _parse_wires(self, content):
        """Parse wire declarations."""
        wire_pattern = r'wire\s+(?:\[.*?\]\s+)?([^;]+);'
        
        for match in re.finditer(wire_pattern, content):
            wires = match.group(1).split(',')
            for wire in wires:
                name = self._normalize_identifier(wire)
                name = re.sub(r'\[.*?\]', '', name).strip()
                if name and name not in self.wires:
                    self.wires.append(name)
    
    def _parse_instances(self, content):
        """Parse gate instances (handles both ICCAD positional and Yosys named formats)."""
        # Pattern to match gate instances:
        # - ICCAD: gate_type ( ports );
        # - Yosys: \gate_type instance_name ( ports );
        # Matches escaped identifiers starting with \ or regular identifiers
        
        # Identifier can be:
        # - Escaped: \$_NAND_ or \a[0] (backslash followed by non-whitespace)
        # - Regular: and, or, _inst123, etc. (word characters)
        instance_pattern = r'(\\[^\s]+|\w+)\s+(?:(\\[^\s]+|\w+)\s+)?\(\s*(.*?)\s*\)\s*;'
        
        for match in re.finditer(instance_pattern, content, re.DOTALL):
            gate_type_raw = match.group(1)
            inst_name = match.group(2) if match.group(2) else 'unnamed'
            port_list = match.group(3)
            
            # Normalize gate type (remove backslash, convert to lowercase)
            gate_type_normalized = self._normalize_identifier(gate_type_raw).lower()
            
            if gate_type_normalized not in self.GATE_MAPPINGS:
                continue
            
            gate_type = self.GATE_MAPPINGS[gate_type_normalized]
            ports = self._parse_port_connections(port_list)
            
            if not ports or len(ports) < 1:
                continue
            
            output = ports[0]
            inputs = ports[1:] if len(ports) > 1 else []
            
            if gate_type == 'DFF':
                self.ppis.append(output)
                if inputs:
                    self.ppos.append(inputs[0])
                    self.dffs.append((output, inputs[0]))
                    self.dff_map[output] = inputs[0]
            else:
                self.gates.append((output, gate_type, inputs))
                self.gate_dict[output] = (gate_type, inputs)
                
                for inp in inputs:
                    if inp not in self.back_edges:
                        self.back_edges[inp] = []
                    self.back_edges[inp].append(output)
    
    def _parse_port_connections(self, port_list):
        """Parse port connections (positional or named)."""
        ports = []
        port_list = port_list.strip()
        
        if '.(' in port_list or ('.' in port_list and '(' in port_list):
            # Named connections - handle escaped identifiers
            # Pattern: .PORT_NAME(\wire_name) or .PORT_NAME(wire_name)
            named_pattern = r'\.(\w+)\s*\(\s*([^)]+)\s*\)'
            connections = {}
            
            for match in re.finditer(named_pattern, port_list):
                port_name = match.group(1)
                wire_name = self._normalize_identifier(match.group(2))
                connections[port_name] = wire_name
            
            # Extract output first
            output_names = ['Y', 'Q', 'OUT', 'Z', 'O']
            for name in output_names:
                if name in connections:
                    ports.append(connections[name])
                    break
            
            # Extract inputs
            skip_ports = ['CLK', 'CLOCK', 'RST', 'RESET', 'SET', 'CLEAR', 'EN', 'ENABLE']
            for port_name, wire_name in connections.items():
                if port_name not in output_names and port_name not in skip_ports:
                    if wire_name not in ports:
                        ports.append(wire_name)
        else:
            # Positional connections
            wires = [self._normalize_identifier(w) for w in port_list.split(',')]
            ports = [w for w in wires if w and w != '1\'b0' and w != '1\'b1']
        
        return ports
    
    # =========================================================================
    # BenchParser-Compatible API
    # =========================================================================
    
    def get_all_wires(self):
        wires = set(self.all_inputs + self.all_outputs + self.wires)
        for out, _, inputs in self.gates:
            wires.add(out)
            wires.update(inputs)
        return sorted(list(wires))
    
    def build_var_map(self):
        if self.var_map:
            return self.var_map
        next_var = 1
        for wire in self.get_all_wires():
            if wire not in self.var_map:
                self.var_map[wire] = next_var
                next_var += 1
        return self.var_map
    
    def get_fanout(self, wire_name):
        return self.back_edges.get(wire_name, [])
    
    def get_fanin(self, wire_name):
        if wire_name in self.gate_dict:
            return self.gate_dict[wire_name][1]
        return []
    
    def is_pi(self, wire_name):
        return wire_name in self.inputs
    
    def is_po(self, wire_name):
        return wire_name in self.outputs
    
    def is_ppi(self, wire_name):
        return wire_name in self.ppis
    
    def is_ppo(self, wire_name):
        return wire_name in self.ppos
    
    def is_dff_output(self, wire_name):
        return wire_name in self.dff_map
    
    def get_dff_input(self, q_output):
        return self.dff_map.get(q_output)
    
    def get_gate_type(self, wire_name):
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