"""
Quick inline test of bar.v to see where it fails
"""
import sys
sys.path.insert(0, '.')

print("Testing bar.v step by step...")
print("="*70)

# Step 1: Parse
print("\n1. PARSING...")
try:
    from VerilogParser_ENHANCED import VerilogParser
    parser = VerilogParser('bar.v')
    print(f"   ✓ Inputs: {len(parser.inputs)}")
    print(f"   ✓ Outputs: {len(parser.outputs)}")  
    print(f"   ✓ Gates: {len(parser.gates)}")
    
    if len(parser.gates) == 0:
        print("\n   ✗✗✗ ZERO GATES! This is the problem!")
        print("   The enhanced parser is NOT being used, or")
        print("   assign statements are not being parsed")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Parsing failed: {e}")
    sys.exit(1)

# Step 2: Create miter
print("\n2. CREATING MITER...")
try:
    from WireFaultMiter import WireFaultMiter
    miter = WireFaultMiter('bar.v')
    print(f"   ✓ Miter created")
except Exception as e:
    print(f"   ✗ Miter failed: {e}")
    sys.exit(1)

# Step 3: Test one fault
print("\n3. TESTING ONE FAULT...")
test_gate = list(parser.gate_dict.keys())[0]
print(f"   Testing fault: {test_gate} SA0")

try:
    reachable = miter.get_reachable_outputs(test_gate)
    if not reachable:
        print(f"   ✗ NO REACHABLE OUTPUTS!")
        print(f"   This is why you get 0 samples!")
        print(f"   The circuit has no path from faults to outputs")
        sys.exit(1)
    print(f"   ✓ Reachable: {len(reachable)} outputs")
except Exception as e:
    print(f"   ✗ get_reachable_outputs failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

target = reachable[0]

try:
    clauses = miter.build_miter(test_gate, 0, force_diff=1, target_output=target)
    if not clauses:
        print(f"   ✗ build_miter returned empty!")
        sys.exit(1)
    print(f"   ✓ Miter built: {len(clauses)} clauses")
except Exception as e:
    print(f"   ✗ build_miter failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    cone = miter.get_complete_atpg_cone(test_gate, target)
    if not cone:
        print(f"   ✗ No cone!")
        sys.exit(1)
    print(f"   ✓ Cone: {len(cone)} gates")
except Exception as e:
    print(f"   ✗ get_complete_atpg_cone failed: {e}")
    sys.exit(1)

try:
    from pysat.solvers import Glucose3
    with Glucose3(bootstrap_with=clauses) as solver:
        solver.conf_budget(10000)
        result = solver.solve()
        if not result:
            print(f"   ✗ UNSAT - fault not testable")
            sys.exit(1)
        print(f"   ✓ SAT - fault is testable!")
except Exception as e:
    print(f"   ✗ SAT solving failed: {e}")
    sys.exit(1)

try:
    cone_inputs = miter.get_cone_inputs(cone)
    if not cone_inputs:
        print(f"   ✗ NO CONE INPUTS!")
        print(f"   This might be why you get 0 samples")
        sys.exit(1)
    print(f"   ✓ Cone inputs: {len(cone_inputs)}")
except Exception as e:
    print(f"   ✗ get_cone_inputs failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL STEPS PASSED!")
print("This fault SHOULD produce a sample!")
print("\nIf you still get 0 samples, the problem is:")
print("1. Different VerilogParser being used (not the enhanced one)")
print("2. Error in data generation loop (check exception handling)")
print("3. All OTHER faults failing (test was lucky)")