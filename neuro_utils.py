"""
Enhanced FastGraphExtractor with full SCOAP metrics:
- Controllability (CC0, CC1): How hard to set a signal to 0 or 1
- Observability (CO): How hard to observe a signal at outputs

Observability is critical for fault detection - it measures how easily
a fault on a wire can propagate to an observable output.
"""
import torch
import math
from torch_geometric.data import Data
from BenchParser import BenchParser

class FastGraphExtractor:
    """
    SCOAP-Complete Graph Extractor with Controllability AND Observability
    """
    
    def __init__(self, bench_path, var_map=None):
        self.parser = BenchParser(bench_path)
        
        if var_map is None:
            var_map = self.parser.build_var_map()
        self.var_map = var_map
        
        self.ordered_names = sorted(var_map.keys(), key=lambda k: var_map[k])
        self.name_to_idx = {name: i for i, name in enumerate(self.ordered_names)}
        self.num_nodes = len(self.ordered_names)
        
        # Build edge list and gate information
        self.edges_list = []
        self.gate_types = {}
        self.gate_inputs = {i: [] for i in range(self.num_nodes)}
        self.gate_outputs = {i: [] for i in range(self.num_nodes)}  # NEW: Track fanout
        
        for out, g_type, inputs in self.parser.gates:
            self.gate_types[out] = g_type
            if out in self.name_to_idx:
                dst = self.name_to_idx[out]
                for inp in inputs:
                    if inp in self.name_to_idx:
                        src = self.name_to_idx[inp]
                        self.edges_list.append([src, dst])
                        self.gate_inputs[dst].append(src)
                        self.gate_outputs[src].append(dst)  # Track fanout
        
        # Mark PPIs (DFF outputs)
        for ppi in self.parser.ppis:
            if ppi in self.name_to_idx:
                self.gate_types[ppi] = 'PPI'
        
        # Mark PIs
        for pi in self.parser.inputs:
            if pi not in self.gate_types and pi in self.name_to_idx:
                self.gate_types[pi] = 'INPUT'
        
        # Build adjacency
        self.adj = {i: [] for i in range(self.num_nodes)}
        self.parents = {i: [] for i in range(self.num_nodes)}
        for src, dst in self.edges_list:
            self.adj[src].append(dst)
            self.adj[dst].append(src)
            self.parents[dst].append(src)

        self.edge_index = torch.tensor(self.edges_list, dtype=torch.long).t().contiguous()
        if self.edge_index.numel() == 0:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
        
        self.x_base = self._build_base_features()
    
    def _calculate_controllability(self):
        """
        Calculate SCOAP Controllability (CC0, CC1)
        
        CC0(wire): Minimum # of input assignments needed to set wire to 0
        CC1(wire): Minimum # of input assignments needed to set wire to 1
        
        Algorithm: Forward propagation from inputs
        """
        cc0 = {i: 1 for i in range(self.num_nodes)}
        cc1 = {i: 1 for i in range(self.num_nodes)}
        
        for _ in range(50):  # Iterate until convergence
            changed = False
            for i in range(self.num_nodes):
                inputs = self.gate_inputs[i]
                if not inputs:
                    continue
                
                name = self.ordered_names[i]
                g_type = self.gate_types.get(name, 'INPUT')
                
                c0s = [cc0[x] for x in inputs]
                c1s = [cc1[x] for x in inputs]
                old_c0, old_c1 = cc0[i], cc1[i]
                new_c0, new_c1 = old_c0, old_c1
                
                # SCOAP Controllability Rules
                if g_type == 'AND':
                    # To get 0: make ANY input 0 (minimum)
                    # To get 1: make ALL inputs 1 (sum)
                    new_c0 = min(c0s) + 1
                    new_c1 = sum(c1s) + 1
                    
                elif g_type == 'OR':
                    # To get 0: make ALL inputs 0 (sum)
                    # To get 1: make ANY input 1 (minimum)
                    new_c0 = sum(c0s) + 1
                    new_c1 = min(c1s) + 1
                    
                elif g_type == 'NAND':
                    # NAND = NOT(AND)
                    new_c0 = sum(c1s) + 1  # To get 0: all inputs 1
                    new_c1 = min(c0s) + 1  # To get 1: any input 0
                    
                elif g_type == 'NOR':
                    # NOR = NOT(OR)
                    new_c0 = min(c1s) + 1  # To get 0: any input 1
                    new_c1 = sum(c0s) + 1  # To get 1: all inputs 0
                    
                elif g_type == 'NOT':
                    # Inverts controllability
                    new_c0 = c1s[0] + 1
                    new_c1 = c0s[0] + 1
                    
                elif g_type == 'BUFF':
                    # Passes through
                    new_c0 = c0s[0] + 1
                    new_c1 = c1s[0] + 1
                    
                elif g_type == 'XOR':
                    # XOR is 1 when inputs differ
                    new_c0 = min(c0s[0] + c0s[1], c1s[0] + c1s[1]) + 1
                    new_c1 = min(c0s[0] + c1s[1], c1s[0] + c0s[1]) + 1
                
                if new_c0 != old_c0 or new_c1 != old_c1:
                    cc0[i] = min(new_c0, 5000)
                    cc1[i] = min(new_c1, 5000)
                    changed = True
            
            if not changed:
                break
        
        return cc0, cc1
    
    def _calculate_observability(self):
        """
        Calculate SCOAP Observability (CO)
        
        CO(wire): Minimum # of additional inputs needed to observe wire at outputs
        
        Algorithm: Backward propagation from outputs
        
        Key insight: A wire is easier to observe if:
        1. It's directly an output (CO = 0)
        2. It feeds gates that are easy to observe
        3. It doesn't require many side inputs to propagate
        """
        co = {i: float('inf') for i in range(self.num_nodes)}
        
        # Initialize outputs with CO = 0
        output_set = set(self.parser.all_outputs)
        for i, name in enumerate(self.ordered_names):
            if name in output_set:
                co[i] = 0
        
        # Get controllability for side input calculations
        cc0, cc1 = self._calculate_controllability()
        
        # Backward propagation (multiple iterations)
        for _ in range(50):
            changed = False
            
            for i in range(self.num_nodes):
                # For each node, check all gates it feeds into
                fanout_gates = self.gate_outputs[i]
                if not fanout_gates:
                    continue
                
                name = self.ordered_names[i]
                
                # Find minimum observability through any fanout path
                min_co = co[i]
                
                for gate_idx in fanout_gates:
                    gate_name = self.ordered_names[gate_idx]
                    g_type = self.gate_types.get(gate_name, 'INPUT')
                    gate_inputs = self.gate_inputs[gate_idx]
                    
                    # CO at this input = CO at gate output + side input controllability
                    gate_co = co[gate_idx]
                    if gate_co == float('inf'):
                        continue
                    
                    # Calculate side input contribution
                    side_input_cost = 0
                    
                    if g_type in ['AND', 'NAND']:
                        # To propagate through AND/NAND: other inputs must be 1
                        for inp_idx in gate_inputs:
                            if inp_idx != i:  # Side inputs
                                side_input_cost += cc1[inp_idx]
                    
                    elif g_type in ['OR', 'NOR']:
                        # To propagate through OR/NOR: other inputs must be 0
                        for inp_idx in gate_inputs:
                            if inp_idx != i:
                                side_input_cost += cc0[inp_idx]
                    
                    elif g_type in ['NOT', 'BUFF']:
                        # No side inputs needed
                        side_input_cost = 0
                    
                    elif g_type == 'XOR':
                        # XOR: need to control other input (either 0 or 1, take min)
                        for inp_idx in gate_inputs:
                            if inp_idx != i:
                                side_input_cost += min(cc0[inp_idx], cc1[inp_idx])
                    
                    # Total observability through this path
                    path_co = gate_co + side_input_cost + 1
                    min_co = min(min_co, path_co)
                
                if min_co < co[i]:
                    co[i] = min(min_co, 5000)
                    changed = True
            
            if not changed:
                break
        
        # Handle unreachable nodes (no path to outputs)
        for i in range(self.num_nodes):
            if co[i] == float('inf'):
                co[i] = 10000  # Very high cost
        
        return co
    
    def _compute_depth(self, reverse=False):
        """Compute topological depth"""
        adj = {i: [] for i in range(self.num_nodes)}
        deg = {i: 0 for i in range(self.num_nodes)}
        
        for src, dst in self.edges_list:
            if reverse:
                adj[dst].append(src)
                deg[src] += 1
            else:
                adj[src].append(dst)
                deg[dst] += 1
        
        q = [i for i in range(self.num_nodes) if deg[i] == 0]
        depth = {i: 0 for i in range(self.num_nodes)}
        
        while q:
            n = q.pop(0)
            for neighbor in adj[n]:
                depth[neighbor] = max(depth[neighbor], depth[n] + 1)
                deg[neighbor] -= 1
                if deg[neighbor] == 0:
                    q.append(neighbor)
        
        max_d = max(depth.values()) if depth else 1
        return [depth[i] / max(max_d, 1) for i in range(self.num_nodes)]
    
    def _build_base_features(self):
        """
        Build base features with FULL SCOAP metrics
        
        Feature dimensions (EXPANDED to 16):
        [0-7]: Gate type one-hot (INPUT, NAND, AND, NOR, OR, NOT, BUFF, XOR)
        [8]: Forward depth (normalized)
        [9]: Backward depth (normalized)
        [10]: Is fault target (set per-fault)
        [11]: Distance from fault (set per-fault)
        [12]: SCOAP CC0 - Controllability to 0 (log normalized)
        [13]: SCOAP CC1 - Controllability to 1 (log normalized)
        [14]: SCOAP CO - Observability (log normalized) ← NEW!
        [15]: Is output node (1.0 if observable) ← NEW!
        """
        types = ['INPUT', 'NAND', 'AND', 'NOR', 'OR', 'NOT', 'BUFF', 'XOR']
        t_map = {t: i for i, t in enumerate(types)}
        
        fwd = self._compute_depth(False)
        rev = self._compute_depth(True)
        cc0, cc1 = self._calculate_controllability()
        co = self._calculate_observability()  # NEW!
        
        # EXPANDED feature matrix: 14 -> 16 dimensions
        x = torch.zeros((self.num_nodes, 16), dtype=torch.float)
        
        output_set = set(self.parser.all_outputs)
        
        for i, name in enumerate(self.ordered_names):
            # Gate type
            g_type = self.gate_types.get(name, 'INPUT')
            if g_type in t_map:
                x[i, t_map[g_type]] = 1.0
            
            # Depth features
            x[i, 8] = fwd[i]
            x[i, 9] = rev[i]
            
            # SCOAP Controllability (log normalized)
            x[i, 12] = math.log(cc0[i] + 1) / 10.0
            x[i, 13] = math.log(cc1[i] + 1) / 10.0
            
            # SCOAP Observability (log normalized) - NEW!
            x[i, 14] = math.log(co[i] + 1) / 10.0
            
            # Output marking - NEW!
            if name in output_set:
                x[i, 15] = 1.0
        
        return x
    
    def get_data_for_fault(self, fault_name):
        """
        Build graph data for a specific fault.
        
        Marks:
        - Fault target wire (feature [10] = 1.0)
        - Distance from fault (feature [11] = normalized distance)
        """
        x = self.x_base.clone()
        
        # Mark fault target
        tid = self.name_to_idx.get(fault_name)
        if tid is not None:
            x[tid, 10] = 1.0
            
            # Compute distance from fault via BFS
            q = [tid]
            dist = {i: -1 for i in range(self.num_nodes)}
            dist[tid] = 0
            
            while q:
                n = q.pop(0)
                if dist[n] >= 10:
                    continue
                for neighbor in self.adj[n]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[n] + 1
                        q.append(neighbor)
            
            # Normalize distances
            max_d = max(dist.values()) if dist else 1
            for i in range(self.num_nodes):
                if dist[i] != -1:
                    x[i, 11] = 1.0 - (dist[i] / max(max_d, 1))
        
        # Create data object
        data = Data(x=x, edge_index=self.edge_index)
        data.node_names = self.ordered_names
        
        return data
    
    def analyze_scoap(self, wire_name):
        """Debug helper: Show SCOAP metrics for a wire"""
        if wire_name not in self.name_to_idx:
            print(f"Wire {wire_name} not found")
            return
        
        idx = self.name_to_idx[wire_name]
        features = self.x_base[idx]
        
        cc0_raw = math.exp(features[12].item() * 10.0) - 1
        cc1_raw = math.exp(features[13].item() * 10.0) - 1
        co_raw = math.exp(features[14].item() * 10.0) - 1
        
        print(f"\n=== SCOAP Analysis for {wire_name} ===")
        print(f"Controllability CC0: {cc0_raw:.1f} (effort to set to 0)")
        print(f"Controllability CC1: {cc1_raw:.1f} (effort to set to 1)")
        print(f"Observability CO:    {co_raw:.1f} (effort to observe at outputs)")
        print(f"Is Output: {features[15].item() == 1.0}")
        
        # Testability interpretation
        total_test_0 = cc0_raw + co_raw
        total_test_1 = cc1_raw + co_raw
        
        print(f"\nTestability (lower = easier to test):")
        print(f"  Stuck-at-0 fault: {total_test_0:.1f}")
        print(f"  Stuck-at-1 fault: {total_test_1:.1f}")
        
        if total_test_0 < 10 and total_test_1 < 10:
            print("  → Easy to test")
        elif total_test_0 > 100 or total_test_1 > 100:
            print("  → Hard to test (may need many patterns)")
        else:
            print("  → Moderate difficulty")


# ==============================================================================
# TEST AND COMPARISON
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    # Test file
    bench_file = "../hdl-benchmarks/iscas89/bench/c1908.bench"
    
    print("="*70)
    print("Testing SCOAP with Observability")
    print("="*70)
    
    extractor = FastGraphExtractor(bench_file)
    
    print(f"\nCircuit loaded:")
    print(f"  Nodes: {extractor.num_nodes}")
    print(f"  Edges: {len(extractor.edges_list)}")
    print(f"  Outputs: {len(extractor.parser.all_outputs)}")
    print(f"  Feature dimensions: {extractor.x_base.shape[1]}")
    
    # Analyze some wires
    print("\n" + "="*70)
    print("SCOAP Analysis Examples")
    print("="*70)
    
    # Analyze an input
    if extractor.parser.inputs:
        input_wire = extractor.parser.inputs[0]
        extractor.analyze_scoap(input_wire)
    
    # Analyze an output
    if extractor.parser.outputs:
        output_wire = extractor.parser.outputs[0]
        extractor.analyze_scoap(output_wire)
    
    # Analyze a random internal gate
    if extractor.parser.gates:
        internal_wire = extractor.parser.gates[len(extractor.parser.gates)//2][0]
        extractor.analyze_scoap(internal_wire)
    
    # Compare observability across all nodes
    print("\n" + "="*70)
    print("Observability Distribution")
    print("="*70)
    
    co_values = []
    for i in range(extractor.num_nodes):
        co_norm = extractor.x_base[i, 14].item()
        co_raw = math.exp(co_norm * 10.0) - 1
        co_values.append(co_raw)
    
    co_values.sort()
    
    print(f"Most observable (easiest to test):")
    for i in range(min(5, len(co_values))):
        print(f"  CO = {co_values[i]:.1f}")
    
    print(f"\nLeast observable (hardest to test):")
    for i in range(max(0, len(co_values)-5), len(co_values)):
        print(f"  CO = {co_values[i]:.1f}")
    
    print(f"\nMedian observability: {co_values[len(co_values)//2]:.1f}")