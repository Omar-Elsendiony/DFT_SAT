"""
Test script to verify BenchParser functionality
"""
from BenchParser import BenchParser
from WireFaultMiter import WireFaultMiter
from neuro_utils import FastGraphExtractor

# Test with a sample ISCAS89 benchmark
BENCH_FILE = "../hdl-benchmarks/iscas89/bench/s298.bench"

def test_parser():
    print("="*70)
    print("Testing BenchParser with", BENCH_FILE)
    print("="*70)
    
    # 1. Test Parser Directly
    parser = BenchParser(BENCH_FILE)
    print(parser)
    print()
    
    print("Primary Inputs (PIs):", parser.inputs[:5], f"... ({len(parser.inputs)} total)")
    print("Primary Outputs (POs):", parser.outputs[:5], f"... ({len(parser.outputs)} total)")
    print("Pseudo Primary Inputs (PPIs/DFF Qs):", parser.ppis[:5], f"... ({len(parser.ppis)} total)")
    print("Pseudo Primary Outputs (PPOs/DFF Ds):", parser.ppos[:5], f"... ({len(parser.ppos)} total)")
    print()
    
    print(f"Total Inputs (PIs + PPIs): {len(parser.all_inputs)}")
    print(f"Total Outputs (POs + PPOs): {len(parser.all_outputs)}")
    print(f"Total Gates: {len(parser.gates)}")
    print()
    
    # Show DFF mappings
    print("DFF Mappings (Q -> D):")
    for i, (q, d) in enumerate(parser.dffs[:5]):
        print(f"  {q} -> {d}")
    if len(parser.dffs) > 5:
        print(f"  ... ({len(parser.dffs)} total)")
    print()
    
    # Test back edges
    print("Back Edge Test (showing fanout of first few inputs):")
    for inp in parser.inputs[:3]:
        fanout = parser.get_fanout(inp)
        print(f"  {inp} drives: {fanout[:3]}" + (f" ... ({len(fanout)} total)" if len(fanout) > 3 else ""))
    print()
    
    # 2. Test WireFaultMiter Integration
    print("-"*70)
    print("Testing WireFaultMiter with shared parser:")
    print("-"*70)
    miter = WireFaultMiter(BENCH_FILE)
    print(f"Miter initialized successfully")
    print(f"  Inputs: {len(miter.inputs)}")
    print(f"  Outputs: {len(miter.outputs)}")
    print(f"  Gates: {len(miter.gates)}")
    print(f"  Scan Inputs (PPIs): {len(miter.scan_inputs)}")
    print(f"  Scan Outputs (PPOs): {len(miter.scan_outputs)}")
    print(f"  Variable Map Size: {len(miter.var_map)}")
    print()
    
    # Test building a miter for a fault
    if miter.gates:
        test_gate = miter.gates[0][0]
        print(f"Building miter for fault on: {test_gate}")
        clauses = miter.build_miter(test_gate, fault_type=0)  # Stuck-at-0
        print(f"  Generated {len(clauses)} CNF clauses")
    print()
    
    # 3. Test FastGraphExtractor Integration
    print("-"*70)
    print("Testing FastGraphExtractor with shared parser:")
    print("-"*70)
    extractor = FastGraphExtractor(BENCH_FILE)
    print(f"Graph Extractor initialized successfully")
    print(f"  Nodes: {extractor.num_nodes}")
    print(f"  Edges: {len(extractor.edges_list)}")
    print(f"  Base Features Shape: {extractor.x_base.shape}")
    print()
    
    # Test getting data for a fault
    if extractor.ordered_names:
        test_node = extractor.ordered_names[10] if len(extractor.ordered_names) > 10 else extractor.ordered_names[0]
        print(f"Getting graph data for fault on: {test_node}")
        data = extractor.get_data_for_fault(test_node)
        print(f"  Node features: {data.x.shape}")
        print(f"  Edge index: {data.edge_index.shape}")
        print(f"  Node names: {len(data.node_names)}")
    print()
    
    print("="*70)
    print("All tests passed successfully!")
    print("="*70)

if __name__ == "__main__":
    test_parser()
