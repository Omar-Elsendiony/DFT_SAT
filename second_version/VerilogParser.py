r"""
VerilogParser - Enhanced with DFF Pattern Detection

CRITICAL ENHANCEMENT:
- Detects DFF outputs by naming patterns when not explicitly instantiated
- Handles synthesized netlists where DFF outputs are in port list
- Patterns: *_reg_Q, *_state_reg*, DFF_*, *_Q, *_dff_*

This fixes circuits like unit01/in_2.v where DFF_state_reg_Q is in the
port list but has no explicit DFF instantiation.
"""

import re

class VerilogParser:
    """Parser for gate-level Verilog with DFF pattern detection."""
    
    GATE_MAPPINGS = {
        # Standard gates (lowercase)
        'and': 'AND', 'or': 'OR', 'not': 'NOT', 'nand': 'NAND',
        'nor': 'NOR', 'xor': 'XOR', 'xnor': 'XNOR', 'buf': 'BUFF',
        'buffer': 'BUFF', 'dff': 'DFF', 'DFF': 'DFF',
        
        # Yosys internal gates
        '$_and_': 'AND', '$_or_': 'OR', '$_not_': 'NOT', '$_nand_': 'NAND',
        '$_nor_': 'NOR', '$_xor_': 'XOR', '$_xnor_': 'XNOR', '$_buf_': 'BUFF',
        '$_dff_': 'DFF', '$_dffe_': 'DFF',
        '$_mux_': 'MUX', '$_nmux_': 'NMUX',
        '$_aoi3_': 'AOI3', '$_oai3_': 'OAI3',
        '$_aoi4_': 'AOI4', '$_oai4_': 'OAI4',
    }
    
    # DFF naming patterns (common in synthesized netlists)
    DFF_OUTPUT_PATTERNS = [
        r'.*_reg_Q$',           # state_reg_Q, data_reg_Q
        r'.*_state_reg.*',      # DFF_state_reg, state_reg_0
        r'^DFF_.*',             # DFF_state_reg_Q, DFF_0
        r'.*_Q$',               # flip_flop_Q, reg_Q
        r'.*_dff_.*',           # my_dff_out, dff_0
        r'.*_ff_.*',            # my_ff_out, ff_0
        r'.*\[Q\]$',            # reg[Q], state[Q]
    ]
    
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
        
        # Track potential DFF outputs found in port list
        self.potential_dff_outputs = set()
        
        self._parse()
    
    def _is_dff_output_name(self, name):
        """Check if wire name matches DFF output patterns."""
        for pattern in self.DFF_OUTPUT_PATTERNS:
            if re.match(pattern, name, re.IGNORECASE):
                return True
        return False
    
    def _remove_comments(self, content):
        """Remove ALL comments including Yosys headers."""
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        return content
    
    def _normalize_identifier(self, name):
        """Normalize Verilog identifiers."""
        name = name.strip()
        if name.startswith('\\'):
            name = name[1:].strip()
        return name
    
    def _parse(self):
        """Parse Verilog file."""
        with open(self.verilog_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        content = self._remove_comments(content)
        modules = self._extract_modules(content)
        
        if not modules:
            raise ValueError("No module found in Verilog file")
        
        module_content = modules[0]
        
        # Parse in order
        self._parse_ports(module_content)
        self._parse_wires(module_content)
        self._parse_instances(module_content)
        self._parse_assign_statements(module_content)  # NEW: Parse assign statements
        
        # NEW: Detect DFF outputs by naming pattern
        self._detect_dff_outputs_by_pattern()
        
        # Build combined lists
        self.all_inputs = list(dict.fromkeys(self.inputs + self.ppis))
        self.all_outputs = list(dict.fromkeys(self.outputs + self.ppos))
    
    def _extract_modules(self, content):
        """Extract module definitions."""
        pattern = r'module\s+(\S+)\s*\((.*?)\);(.*?)endmodule'
        matches = re.findall(pattern, content, re.DOTALL)
        return [match[2] for match in matches]
    
    def _parse_ports(self, content):
        """Parse input/output declarations and detect potential DFF outputs."""
        input_pattern = r'input\s+(?:\[.*?\]\s+)?([^;]+);'
        output_pattern = r'output\s+(?:\[.*?\]\s+)?([^;]+);'
        
        # Use re.DOTALL to handle multi-line declarations
        for match in re.finditer(input_pattern, content, re.DOTALL):
            ports = match.group(1).split(',')
            for port in ports:
                name = self._normalize_identifier(port)
                # Don't strip array indices - keep them!
                # name = re.sub(r'\[.*?\]', '', name).strip()  # OLD - strips [0], [1]
                if name and name not in self.inputs:
                    self.inputs.append(name)
                    
                    # Check if this looks like a DFF output
                    if self._is_dff_output_name(name):
                        self.potential_dff_outputs.add(name)
        
        # Use re.DOTALL to handle multi-line declarations
        for match in re.finditer(output_pattern, content, re.DOTALL):
            ports = match.group(1).split(',')
            for port in ports:
                name = self._normalize_identifier(port)
                # Don't strip array indices - keep them!
                # name = re.sub(r'\[.*?\]', '', name).strip()  # OLD - strips [0], [1]
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
                    
                    # Check if this looks like a DFF output
                    if self._is_dff_output_name(name):
                        self.potential_dff_outputs.add(name)
    
    def _parse_instances(self, content):
        """Parse gate instances."""
        instance_pattern = r'(\\[^\s]+|\w+)\s+(?:(\\[^\s]+|\w+)\s+)?\(\s*(.*?)\s*\)\s*;'
        
        for match in re.finditer(instance_pattern, content, re.DOTALL):
            gate_type_raw = match.group(1)
            inst_name = match.group(2) if match.group(2) else 'unnamed'
            port_list = match.group(3)
            
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
                # Remove from potential DFFs (already found explicitly)
                self.potential_dff_outputs.discard(output)
            else:
                self.gates.append((output, gate_type, inputs))
                self.gate_dict[output] = (gate_type, inputs)
                
                for inp in inputs:
                    if inp not in self.back_edges:
                        self.back_edges[inp] = []
                    self.back_edges[inp].append(output)
    
    def _detect_dff_outputs_by_pattern(self):
        """
        NEW: Detect DFF outputs that weren't explicitly instantiated.
        
        This handles synthesized netlists where DFF outputs appear in:
        - Module port list (as inputs)
        - Wire declarations
        But no explicit DFF gate instantiation exists.
        """
        for wire_name in self.potential_dff_outputs:
            # Skip if already identified as DFF from instance
            if wire_name in self.ppis:
                continue
            
            # Check if this wire is actually used in the circuit
            # (has fanout or is in gate connections)
            is_used = (
                wire_name in self.back_edges or
                wire_name in self.gate_dict or
                any(wire_name in inputs for _, _, inputs in self.gates)
            )
            
            if is_used:
                # Add as PPI
                if wire_name in self.inputs:
                    self.inputs.remove(wire_name)
                
                if wire_name not in self.ppis:
                    self.ppis.append(wire_name)
                
                # Try to find corresponding D input (wire driving this DFF)
                # Look for pattern: wire_name_D, wire_name without _Q, etc.
                d_candidates = [
                    wire_name.replace('_Q', '_D'),
                    wire_name.replace('_Q', ''),
                    wire_name + '_D',
                    wire_name.replace('_reg_Q', '_reg_D'),
                ]
                
                for d_wire in d_candidates:
                    if d_wire in self.gate_dict or d_wire in self.wires:
                        if d_wire not in self.ppos:
                            self.ppos.append(d_wire)
                        self.dffs.append((wire_name, d_wire))
                        self.dff_map[wire_name] = d_wire
                        break
                else:
                    # No D input found, just mark Q as PPI without PPO
                    # (This is OK for ATPG - we can set Q to any value)
                    pass
                
                # Removed print to avoid flooding output in multiprocessing
    
    def _parse_port_connections(self, port_list):
        """Parse port connections (positional or named)."""
        ports = []
        port_list = port_list.strip()
        
        if '.(' in port_list or ('.' in port_list and '(' in port_list):
            # Named connections
            named_pattern = r'\.(\w+)\s*\(\s*([^)]+)\s*\)'
            connections = {}
            
            for match in re.finditer(named_pattern, port_list):
                port_name = match.group(1)
                wire_name = self._normalize_identifier(match.group(2))
                connections[port_name] = wire_name
            
            output_names = ['Y', 'Q', 'OUT', 'Z', 'O']
            for name in output_names:
                if name in connections:
                    ports.append(connections[name])
                    break
            
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
    
    def _parse_assign_statements(self, content):
        """
        Parse Verilog assign statements and convert to gate instances.
        
        Handles expressions like:
            assign out = a & b;
            assign out = ~a;
            assign out = a | b | c;
        """
        # Pattern: assign output = expression;
        assign_pattern = r'assign\s+([^\s=]+)\s*=\s*([^;]+);'
        
        for match in re.finditer(assign_pattern, content, re.MULTILINE):
            output = self._normalize_identifier(match.group(1))
            expression = match.group(2).strip()
            
            # Parse the expression and convert to gates
            gate_info = self._parse_expression(expression)
            
            if gate_info:
                gate_type, inputs = gate_info
                self.gates.append((output, gate_type, inputs))
                self.gate_dict[output] = (gate_type, inputs)
                
                for inp in inputs:
                    if inp not in self.back_edges:
                        self.back_edges[inp] = []
                    self.back_edges[inp].append(output)
    
    def _parse_expression(self, expr):
        """
        Parse a Verilog expression and return (gate_type, inputs).
        
        Supports:
            a & b           → AND, [a, b]
            a | b           → OR, [a, b]
            a ^ b           → XOR, [a, b]
            ~a              → NOT, [a]
            ~(a & b)        → NAND, [a, b]
            ~(a | b)        → NOR, [a, b]
            a & b & c       → AND, [a, b, c]
        """
        expr = expr.strip()
        
        # Check for NOT (single input with ~)
        if expr.startswith('~') and '&' not in expr and '|' not in expr and '^' not in expr:
            # Simple NOT: ~a
            inner = self._normalize_identifier(expr[1:].strip())
            return ('NOT', [inner])
        
        # Check for NAND: ~(a & b)
        if expr.startswith('~(') and expr.endswith(')'):
            inner_expr = expr[2:-1].strip()
            if '&' in inner_expr and '|' not in inner_expr:
                inputs = self._extract_operands(inner_expr, '&')
                return ('NAND', inputs)
            elif '|' in inner_expr and '&' not in inner_expr:
                inputs = self._extract_operands(inner_expr, '|')
                return ('NOR', inputs)
        
        # Check for AND: a & b
        if '&' in expr and '|' not in expr:
            inputs = self._extract_operands(expr, '&')
            return ('AND', inputs)
        
        # Check for OR: a | b
        if '|' in expr and '&' not in expr:
            inputs = self._extract_operands(expr, '|')
            return ('OR', inputs)
        
        # Check for XOR: a ^ b
        if '^' in expr:
            inputs = self._extract_operands(expr, '^')
            return ('XOR', inputs)
        
        # Direct wire assignment: assign out = in;
        # This is a buffer
        wire = self._normalize_identifier(expr)
        if wire:
            return ('BUFF', [wire])
        
        return None
    
    def _extract_operands(self, expr, operator):
        """
        Extract operands from an expression split by an operator.
        
        Handles negations: a & ~b → [a, ~b]
        But normalizes wire names properly.
        """
        parts = expr.split(operator)
        operands = []
        
        for part in parts:
            part = part.strip()
            
            # Handle negation
            if part.startswith('~'):
                wire = self._normalize_identifier(part[1:].strip())
                # For now, we'll treat ~input as a separate wire
                # The NOT will be implicit in the gate
                # This is a simplification but works for most cases
                operands.append(wire)
            else:
                wire = self._normalize_identifier(part)
                operands.append(wire)
        
        return [op for op in operands if op]
    
    # =========================================================================
    # BenchParser-Compatible API
    # =========================================================================
    
    def get_all_wires(self):
        """Get all wires actually used in the circuit."""
        wires = set(self.all_inputs + self.all_outputs)
        
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