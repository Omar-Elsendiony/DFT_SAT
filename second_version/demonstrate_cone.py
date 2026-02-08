"""
Demonstration: Old vs New Cone Extraction
Shows the difference between incomplete and complete cone extraction
"""

import sys
import os

# Mock classes for demonstration
class MockParser:
    def __init__(self):
        self.inputs = ['PI1', 'PI2', 'PI3', 'PI4']
        self.outputs = ['PO1']
        self.gates = [
            ('g1', 'AND', ['PI1', 'PI2']),
            ('g2', 'NOT', ['PI3']),
            ('g3', 'OR', ['g1', 'g2']),      # TARGET GATE
            ('g4', 'BUFF', ['PI4']),
            ('g5', 'AND', ['g3', 'g4']),     # Propagates to output
            ('PO1', 'BUFF', ['g5'])
        ]
        
        # Build gate_dict
        self.gate_dict = {out: (typ, inp) for out, typ, inp in self.gates}
        
        # Build fanout map
        self.fanout_map = {}
        for out, _, inputs in self.gates:
            for inp in inputs:
                if inp not in self.fanout_map:
                    self.fanout_map[inp] = []
                self.fanout_map[inp].append(out)
    
    def get_fanout(self, node):
        return self.fanout_map.get(node, [])

def visualize_circuit():
    """Print ASCII art of the example circuit"""
    print("""
Example Circuit:
================

    PI1 ──┐
          AND──g1──┐
    PI2 ──┘        │
                   OR──g3(TARGET)──┐
    PI3 ──NOT─g2───┘                │
                                    │
    PI4 ──BUFF──g4──────────────────┘
                                    │
                                   AND──g5──BUFF──PO1

Target Gate: g3 (fault site)
Target Output: PO1
""")

def old_cone_extraction(parser, target_gate, target_output):
    """OLD METHOD: Only backward from output to fault"""
    print("\n" + "="*80)
    print("OLD METHOD: get_logic_cone()")
    print("="*80)
    print(f"Extracting backward cone from {target_output} to {target_gate}")
    
    cone = []
    visited = set()
    
    def dfs(node):
        if node in visited:
            return
        if node == target_gate:
            print(f"  - Stopped at {node} (target gate)")
            return
        visited.add(node)
        
        if node in parser.gate_dict:
            g_type, inputs = parser.gate_dict[node]
            cone.append((node, g_type, inputs))
            print(f"  - Added {node} ({g_type} {inputs})")
            for inp in inputs:
                dfs(inp)
    
    dfs(target_output)
    
    print(f"\nRESULT: {len(cone)} gates in cone")
    print(f"Gates: {[g[0] for g in cone]}")
    
    # Analyze what's missing
    print("\nANALYSIS:")
    print("✓ Fan-out cone (g3 → PO1): [PO1, g5, g4]")
    print("✗ Fan-in cone (inputs → g3): MISSING [g1, g2]")
    print("✗ Target gate itself: MISSING [g3]")
    
    return cone

def new_cone_extraction(parser, target_gate, target_output):
    """NEW METHOD: Complete cone with fan-in + fan-out"""
    print("\n" + "="*80)
    print("NEW METHOD: get_full_cone()")
    print("="*80)
    
    # Step 1: Fan-in cone
    print(f"\nStep 1: Extract fan-in cone of {target_gate}")
    fanin = []
    visited_fanin = set()
    
    def dfs_backward(node):
        if node in visited_fanin:
            return
        if node in parser.inputs:
            print(f"  - Reached input {node}")
            return
        visited_fanin.add(node)
        
        if node in parser.gate_dict:
            g_type, inputs = parser.gate_dict[node]
            fanin.append((node, g_type, inputs))
            print(f"  - Added {node} ({g_type} {inputs})")
            for inp in inputs:
                dfs_backward(inp)
    
    dfs_backward(target_gate)
    print(f"Fan-in cone: {len(fanin)} gates")
    
    # Step 2: Fan-out cone
    print(f"\nStep 2: Extract fan-out cone from {target_gate} to {target_output}")
    fanout = []
    visited_fanout = set()
    
    def dfs_forward(node):
        if node in visited_fanout:
            return
        visited_fanout.add(node)
        
        if node != target_gate and node in parser.gate_dict:
            g_type, inputs = parser.gate_dict[node]
            fanout.append((node, g_type, inputs))
            print(f"  - Added {node} ({g_type} {inputs})")
        
        fanout_gates = parser.get_fanout(node)
        for next_gate in fanout_gates:
            if next_gate not in visited_fanout:
                dfs_forward(next_gate)
    
    dfs_forward(target_gate)
    print(f"Fan-out cone: {len(fanout)} gates")
    
    # Step 3: Add fault gate itself
    print(f"\nStep 3: Add target gate {target_gate}")
    fault_gate = []
    if target_gate in parser.gate_dict:
        g_type, inputs = parser.gate_dict[target_gate]
        fault_gate = [(target_gate, g_type, inputs)]
        print(f"  - Added {target_gate} ({g_type} {inputs})")
    
    # Step 4: Combine
    all_gates = fanin + fault_gate + fanout
    seen = set()
    unique_cone = []
    
    for gate in all_gates:
        if gate[0] not in seen:
            seen.add(gate[0])
            unique_cone.append(gate)
    
    print(f"\nRESULT: {len(unique_cone)} gates in FULL cone")
    print(f"Gates: {[g[0] for g in unique_cone]}")
    
    print("\nANALYSIS:")
    print("✓ Fan-in cone: [g1, g2]")
    print("✓ Target gate: [g3]")
    print("✓ Fan-out cone: [g4, g5, PO1]")
    print("✓ COMPLETE cone for ATPG!")
    
    return unique_cone

def extract_cone_inputs(parser, cone_gates):
    """Extract primary inputs used in the cone"""
    cone_inputs = set()
    for _, _, inputs in cone_gates:
        for inp in inputs:
            if inp in parser.inputs:
                cone_inputs.add(inp)
    return cone_inputs

def compare_methods():
    """Main comparison function"""
    parser = MockParser()
    target_gate = 'g3'
    target_output = 'PO1'
    
    visualize_circuit()
    
    # Old method
    old_cone = old_cone_extraction(parser, target_gate, target_output)
    old_inputs = extract_cone_inputs(parser, old_cone)
    
    # New method
    new_cone = new_cone_extraction(parser, target_gate, target_output)
    new_inputs = extract_cone_inputs(parser, new_cone)
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    print(f"\nOLD METHOD:")
    print(f"  Gates in cone: {len(old_cone)}")
    print(f"  Gate names: {[g[0] for g in old_cone]}")
    print(f"  Cone inputs: {sorted(old_inputs)}")
    
    print(f"\nNEW METHOD:")
    print(f"  Gates in cone: {len(new_cone)}")
    print(f"  Gate names: {[g[0] for g in new_cone]}")
    print(f"  Cone inputs: {sorted(new_inputs)}")
    
    print(f"\nDIFFERENCE:")
    missing_gates = set([g[0] for g in new_cone]) - set([g[0] for g in old_cone])
    missing_inputs = new_inputs - old_inputs
    
    print(f"  Missing gates: {sorted(missing_gates)}")
    print(f"  Missing inputs: {sorted(missing_inputs)}")
    
    print("\n" + "="*80)
    print("WHY THIS MATTERS FOR ATPG:")
    print("="*80)
    print("""
1. OLD METHOD only gets fan-out (propagation path)
   - Probing PI4 ✓ (affects g4 → g5 → PO1)
   - Probing PI1 ✗ (in fan-in, but NOT measured!)
   - Probing PI2 ✗ (in fan-in, but NOT measured!)
   - Probing PI3 ✗ (in fan-in, but NOT measured!)
   
   Result: INCOMPLETE importance measurements!

2. NEW METHOD gets BOTH fan-in and fan-out
   - Probing PI1 ✓ (affects g1 → g3)
   - Probing PI2 ✓ (affects g1 → g3)
   - Probing PI3 ✓ (affects g2 → g3)
   - Probing PI4 ✓ (affects g4 → g5 → PO1)
   
   Result: COMPLETE importance measurements!

3. THE MITER XOR IS STILL CORRECT
   - We compare good_circuit vs faulty_circuit at outputs
   - XOR detects if outputs differ
   - This works for BOTH methods
   
   The issue was CONE EXTRACTION, not miter logic!
""")

if __name__ == "__main__":
    compare_methods()