import torch
from torch_geometric.data import Data
from BenchParser import BenchParser
import math

# --- FAST GRAPH EXTRACTOR (With SCOAP) ---
class FastGraphExtractor:
    def __init__(self, bench_path, var_map=None):
        # Use shared parser
        self.parser = BenchParser(bench_path)
        
        # Use provided var_map or build from parser
        if var_map is None:
            var_map = self.parser.build_var_map()
        self.var_map = var_map
        
        # Create ordered list of node names and indices
        self.ordered_names = sorted(var_map.keys(), key=lambda k: var_map[k])
        self.name_to_idx = {name: i for i, name in enumerate(self.ordered_names)}
        self.num_nodes = len(self.ordered_names)
        
        # Build edge list and gate information
        self.edges_list = []
        self.gate_types = {}
        self.gate_inputs = {i: [] for i in range(self.num_nodes)}
        
        # Process gates from parser
        for out, g_type, inputs in self.parser.gates:
            self.gate_types[out] = g_type
            if out in self.name_to_idx:
                dst = self.name_to_idx[out]
                for inp in inputs:
                    if inp in self.name_to_idx:
                        src = self.name_to_idx[inp]
                        self.edges_list.append([src, dst])
                        self.gate_inputs[dst].append(src)
        
        # Mark PPIs (DFF outputs) as special inputs
        for ppi in self.parser.ppis:
            if ppi in self.name_to_idx:
                self.gate_types[ppi] = 'PPI'
        
        # Mark PIs
        for pi in self.parser.inputs:
            if pi not in self.gate_types and pi in self.name_to_idx:
                self.gate_types[pi] = 'INPUT'
        
        # Build adjacency structures (including back edges from parser)
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

    def _calculate_scoap(self):
        cc0 = {i: 1 for i in range(self.num_nodes)}
        cc1 = {i: 1 for i in range(self.num_nodes)}
        for _ in range(50):
            changed = False
            for i in range(self.num_nodes):
                inputs = self.gate_inputs[i]
                if not inputs: continue
                name = self.ordered_names[i]
                g_type = self.gate_types.get(name, 'INPUT')
                
                c0s = [cc0[x] for x in inputs]
                c1s = [cc1[x] for x in inputs]
                old_c0, old_c1 = cc0[i], cc1[i]
                new_c0, new_c1 = old_c0, old_c1
                
                if g_type == 'AND': new_c0, new_c1 = min(c0s)+1, sum(c1s)+1
                elif g_type == 'OR': new_c0, new_c1 = sum(c0s)+1, min(c1s)+1
                elif g_type == 'NAND': new_c0, new_c1 = sum(c1s)+1, min(c0s)+1
                elif g_type == 'NOR': new_c0, new_c1 = min(c1s)+1, sum(c0s)+1
                elif g_type == 'NOT': new_c0, new_c1 = c1s[0]+1, c0s[0]+1
                elif g_type == 'BUFF': new_c0, new_c1 = c0s[0]+1, c1s[0]+1
                elif g_type == 'XOR': 
                    new_c0 = min(c0s[0]+c0s[1], c1s[0]+c1s[1]) + 1
                    new_c1 = min(c0s[0]+c1s[1], c1s[0]+c0s[1]) + 1
                
                if new_c0 != old_c0 or new_c1 != old_c1:
                    cc0[i], cc1[i] = min(new_c0, 5000), min(new_c1, 5000)
                    changed = True
            if not changed: break
        return cc0, cc1

    def _compute_depth(self, reverse=False):
        adj = {i: [] for i in range(self.num_nodes)}
        deg = {i: 0 for i in range(self.num_nodes)}
        for src, dst in self.edges_list:
            if reverse: adj[dst].append(src); deg[src] += 1
            else: adj[src].append(dst); deg[dst] += 1
        q = [i for i in range(self.num_nodes) if deg[i] == 0]
        depth = {i: 0 for i in range(self.num_nodes)}
        while q:
            n = q.pop(0)
            for neighbor in adj[n]:
                depth[neighbor] = max(depth[neighbor], depth[n]+1)
                deg[neighbor] -= 1
                if deg[neighbor] == 0: q.append(neighbor)
        max_d = max(depth.values()) if depth else 1
        return [depth[i]/max(max_d, 1) for i in range(self.num_nodes)]

    def _build_base_features(self):
        types = ['INPUT', 'NAND', 'AND', 'NOR', 'OR', 'NOT', 'BUFF', 'XOR']
        t_map = {t: i for i, t in enumerate(types)}
        fwd = self._compute_depth(False)
        rev = self._compute_depth(True)
        cc0, cc1 = self._calculate_scoap()
        
        x = torch.zeros((self.num_nodes, 14), dtype=torch.float)
        for i, name in enumerate(self.ordered_names):
            g_type = self.gate_types.get(name, 'INPUT')
            if g_type in t_map: x[i, t_map[g_type]] = 1.0
            x[i, 8] = fwd[i]
            x[i, 9] = rev[i]
            x[i, 12] = math.log(cc0[i] + 1) / 10.0
            x[i, 13] = math.log(cc1[i] + 1) / 10.0
        return x

    def get_data_for_fault(self, fault_name):
        x = self.x_base.clone()
        tid = self.name_to_idx.get(fault_name)
        if tid is not None:
            x[tid, 10] = 1.0
            q = [tid]
            dist = {i: -1 for i in range(self.num_nodes)}; dist[tid] = 0
            while q:
                n = q.pop(0)
                if dist[n] >= 10: continue
                for neighbor in self.adj[n]:
                    if dist[neighbor] == -1: dist[neighbor] = dist[n]+1; q.append(neighbor)
            max_d = max(dist.values()) if dist else 1
            for i in range(self.num_nodes):
                if dist[i] != -1: x[i, 11] = 1.0 - (dist[i]/max(max_d, 1))
        
        data = Data(x=x, edge_index=self.edge_index)
        data.node_names = self.ordered_names
        return data