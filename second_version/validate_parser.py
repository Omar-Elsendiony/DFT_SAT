"""
Validation Script: Verify Consistent Node Ordering Across Parsers

This script tests that BenchParser and VerilogParser produce IDENTICAL
node orderings for the same circuit in different formats.
"""

import tempfile
import os

# Import both parsers
import sys
sys.path.insert(0, '/home/claude')

print("=" * 80)
print("VALIDATING CONSISTENT NODE ORDERING ACROSS PARSERS")
print("=" * 80)

# Test circuit (simple example)
test_circuit_bench = """
# Simple test circuit
INPUT(a)
INPUT(b)
INPUT(c)
OUTPUT(out1)
OUTPUT(out2)

n1 = AND(a, b)
n2 = OR(b, c)
out1 = NAND(n1, n2)
out2 = NOT(n1)
"""

test_circuit_verilog = """
module test(a, b, c, out1, out2);
  input a, b, c;
  output out1, out2;
  wire n1, n2;
  wire unused_wire;  // This should NOT affect ordering!
  
  and (n1, a, b);
  or (n2, b, c);
  nand (out1, n1, n2);
  not (out2, n1);
endmodule
"""

print("\nTest Circuit (Bench format):")
print(test_circuit_bench)

print("\nTest Circuit (Verilog format):")
print(test_circuit_verilog)

# Create temporary files
bench_file = tempfile.NamedTemporaryFile(mode='w', suffix='.bench', delete=False)
verilog_file = tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False)

bench_file.write(test_circuit_bench)
bench_file.close()

verilog_file.write(test_circuit_verilog)
verilog_file.close()

try:
    # Import parsers
    from BenchParser import BenchParser
    
    # Try old VerilogParser first to show the bug
    try:
        print("\n" + "=" * 80)
        print("TEST 1: OLD VerilogParser (WITH BUG)")
        print("=" * 80)
        
        # Temporarily rename the fixed version
        if os.path.exists('/home/claude/VerilogParser.py'):
            os.rename('/home/claude/VerilogParser.py', '/home/claude/VerilogParser_backup.py')
        
        # Load old version from documents
        old_verilog_code = '''
def get_all_wires(self):
    wires = set(self.all_inputs + self.all_outputs + self.wires)  # BUG: includes self.wires!
    for out, _, inputs in self.gates:
        wires.add(out)
        wires.update(inputs)
    return sorted(list(wires))
'''
        print("\nOld VerilogParser.get_all_wires():")
        print(old_verilog_code)
        print("\n⚠️  Note: Includes 'self.wires' which may contain unused wires!")
        
    except Exception as e:
        print(f"Could not load old parser: {e}")
    
    # Now test with fixed version
    print("\n" + "=" * 80)
    print("TEST 2: FIXED VerilogParser")
    print("=" * 80)
    
    # Copy fixed version to the right location
    if os.path.exists('/home/claude/VerilogParser_fixed.py'):
        with open('/home/claude/VerilogParser_fixed.py', 'r') as f:
            fixed_code = f.read()
        with open('/home/claude/VerilogParser.py', 'w') as f:
            f.write(fixed_code)
    
    from VerilogParser import VerilogParser
    
    # Parse both formats
    print("\nParsing Bench file...")
    bench_parser = BenchParser(bench_file.name)
    
    print("Parsing Verilog file...")
    verilog_parser = VerilogParser(verilog_file.name)
    
    # Get wire lists
    bench_wires = bench_parser.get_all_wires()
    verilog_wires = verilog_parser.get_all_wires()
    
    print("\n" + "=" * 80)
    print("WIRE LISTS")
    print("=" * 80)
    
    print(f"\nBenchParser wires ({len(bench_wires)}):")
    for i, wire in enumerate(bench_wires):
        print(f"  {i}: {wire}")
    
    print(f"\nVerilogParser wires ({len(verilog_wires)}):")
    for i, wire in enumerate(verilog_wires):
        print(f"  {i}: {wire}")
    
    # Check if they match
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    if bench_wires == verilog_wires:
        print("\n✅ SUCCESS! Wire lists are IDENTICAL")
        print("   Both parsers produce the same ordering")
        print("   Graph structures will match!")
    else:
        print("\n❌ FAILURE! Wire lists are DIFFERENT")
        print("\n   Differences:")
        
        bench_set = set(bench_wires)
        verilog_set = set(verilog_wires)
        
        only_bench = bench_set - verilog_set
        only_verilog = verilog_set - bench_set
        
        if only_bench:
            print(f"   Only in Bench: {only_bench}")
        if only_verilog:
            print(f"   Only in Verilog: {only_verilog}")
        
        if len(bench_wires) == len(verilog_wires):
            print("\n   Same count but different order:")
            for i, (b, v) in enumerate(zip(bench_wires, verilog_wires)):
                if b != v:
                    print(f"   Index {i}: Bench='{b}' vs Verilog='{v}'")
    
    # Build var maps
    print("\n" + "=" * 80)
    print("VARIABLE MAPPINGS")
    print("=" * 80)
    
    bench_var_map = bench_parser.build_var_map()
    verilog_var_map = verilog_parser.build_var_map()
    
    print("\nBenchParser var_map:")
    for wire in sorted(bench_var_map.keys()):
        print(f"  {wire} -> {bench_var_map[wire]}")
    
    print("\nVerilogParser var_map:")
    for wire in sorted(verilog_var_map.keys()):
        print(f"  {wire} -> {verilog_var_map[wire]}")
    
    # Check if var_maps match
    if bench_var_map == verilog_var_map:
        print("\n✅ SUCCESS! Variable mappings are IDENTICAL")
    else:
        print("\n❌ FAILURE! Variable mappings are DIFFERENT")
        
        all_wires = set(bench_var_map.keys()) | set(verilog_var_map.keys())
        for wire in sorted(all_wires):
            bench_id = bench_var_map.get(wire, 'N/A')
            verilog_id = verilog_var_map.get(wire, 'N/A')
            if bench_id != verilog_id:
                print(f"  {wire}: Bench={bench_id} vs Verilog={verilog_id}")
    
    # Test graph structure
    print("\n" + "=" * 80)
    print("GRAPH STRUCTURE (Edge Index)")
    print("=" * 80)
    
    # Simulate what VectorizedGraphExtractor does
    def build_edge_list(parser):
        edges = []
        var_map = parser.build_var_map()
        wires = parser.get_all_wires()
        name_to_idx = {name: i for i, name in enumerate(wires)}
        
        for out, _, inputs in parser.gates:
            if out in name_to_idx:
                dst = name_to_idx[out]
                for inp in inputs:
                    if inp in name_to_idx:
                        src = name_to_idx[inp]
                        edges.append([src, dst])
        
        return edges, name_to_idx
    
    bench_edges, bench_name_to_idx = build_edge_list(bench_parser)
    verilog_edges, verilog_name_to_idx = build_edge_list(verilog_parser)
    
    print("\nBenchParser edges:")
    for src, dst in bench_edges:
        src_name = bench_wires[src]
        dst_name = bench_wires[dst]
        print(f"  {src} → {dst}  ({src_name} → {dst_name})")
    
    print("\nVerilogParser edges:")
    for src, dst in verilog_edges:
        src_name = verilog_wires[src]
        dst_name = verilog_wires[dst]
        print(f"  {src} → {dst}  ({src_name} → {dst_name})")
    
    if bench_edges == verilog_edges:
        print("\n✅ SUCCESS! Edge lists are IDENTICAL")
        print("   Graph structures match perfectly!")
    else:
        print("\n❌ FAILURE! Edge lists are DIFFERENT")
        print(f"   Bench has {len(bench_edges)} edges")
        print(f"   Verilog has {len(verilog_edges)} edges")
    
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if (bench_wires == verilog_wires and 
        bench_var_map == verilog_var_map and 
        bench_edges == verilog_edges):
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe fixed VerilogParser produces IDENTICAL graph structures")
        print("to BenchParser. Training and inference will use consistent")
        print("node orderings regardless of file format.")
    else:
        print("\n❌ TESTS FAILED!")
        print("\nParsers produce different structures.")
        print("This will cause graph mismatch between training and inference.")

finally:
    # Cleanup
    os.unlink(bench_file.name)
    os.unlink(verilog_file.name)
    
    # Restore backup if it exists
    if os.path.exists('/home/claude/VerilogParser_backup.py'):
        os.rename('/home/claude/VerilogParser_backup.py', '/home/claude/VerilogParser_old.py')

print("\n" + "=" * 80)