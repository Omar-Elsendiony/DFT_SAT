"""
Unified Circuit Parser

Auto-detects file format (.bench or .v) and uses appropriate parser.
Provides a single interface for both formats.
"""

import os
from BenchParser import BenchParser
from VerilogParser import VerilogParser

class UnifiedParser:
    """
    Wrapper that auto-detects and delegates to BenchParser or VerilogParser.
    Provides identical API regardless of input format.
    """
    
    def __init__(self, circuit_file):
        self.circuit_file = circuit_file
        
        # Detect format and create appropriate parser
        if circuit_file.endswith('.bench'):
            self.parser = BenchParser(circuit_file)
            self.format = 'bench'
        elif circuit_file.endswith('.v') or circuit_file.endswith('.verilog'):
            self.parser = VerilogParser(circuit_file)
            self.format = 'verilog'
        else:
            # Try to detect by content
            self.format = self._detect_format_by_content(circuit_file)
            if self.format == 'bench':
                self.parser = BenchParser(circuit_file)
            elif self.format == 'verilog':
                self.parser = VerilogParser(circuit_file)
            else:
                raise ValueError(f"Unknown circuit format: {circuit_file}")
    
    def _detect_format_by_content(self, filepath):
        """Detect format by examining file content."""
        with open(filepath, 'r') as f:
            content = f.read(1000)  # Read first 1KB
        
        # Check for Verilog keywords
        if 'module' in content and 'endmodule' in content:
            return 'verilog'
        
        # Check for BENCH keywords
        if 'INPUT(' in content or 'OUTPUT(' in content:
            return 'bench'
        
        return None
    
    # Delegate all methods to underlying parser
    def __getattr__(self, name):
        return getattr(self.parser, name)
    
    def __repr__(self):
        return f"UnifiedParser({self.format}): {self.parser}"


# =============================================================================
# Usage Example: Works with BOTH formats transparently
# =============================================================================

if __name__ == "__main__":
    # Works with .bench files
    bench_parser = UnifiedParser("c17.bench")
    print(bench_parser)
    print("Gates:", len(bench_parser.gates))
    
    # Works with .v files  
    verilog_parser = UnifiedParser("circuit.v")
    print(verilog_parser)
    print("Gates:", len(verilog_parser.gates))
    
    # Same API for both!
    for parser in [bench_parser, verilog_parser]:
        print(f"\nInputs: {parser.inputs}")
        print(f"Outputs: {parser.outputs}")
        print(f"Gate count: {len(parser.gates)}")
        var_map = parser.build_var_map()
        print(f"Variables: {len(var_map)}")