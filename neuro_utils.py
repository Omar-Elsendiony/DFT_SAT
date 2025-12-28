import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import math

# --- 1. THE ADVANCED GNN MODEL ---
class CircuitGNN_Advanced(torch.nn.Module):
    def __init__(self, num_node_features=14, num_layers=8, hidden_dim=64, dropout=0.2):
        super(CircuitGNN_Advanced, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Input Layer
        self.convs.append(GATv2Conv(num_node_features, hidden_dim, heads=2, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden Layers (Residuals)
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
            
        # Output Layer
        self.convs.append(GATv2Conv(hidden_dim, 32, heads=2, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(32))
        
        self.classifier = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.convs[0](x, edge_index)
        x = self.bns[0](x)
        x = F.elu(x)
        
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = x + identity # Residual Connection
            
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = F.elu(x)
        
        return self.classifier(x) # Return Logits (No Sigmoid)

# --- 2. FAST GRAPH EXTRACTOR (With SCOAP) ---
class FastGraphExtractor:
    def __init__(self, bench_path, var_map):
        self.var_map = var_map
        self.ordered_names = sorted(var_map.keys(), key=lambda k: var_map[k])
        self.name_to_idx = {name: i for i, name in enumerate(self.ordered_names)}
        self.num_nodes = len(self.ordered_names)
        
        self.edges_list = []
        self.gate_types = {}
        self.gate_inputs = {i: [] for i in range(self.num_nodes)}
        
        with open(bench_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if '=' in line and not line.startswith('INPUT') and not line.startswith('OUTPUT'):
                    parts = line.split('=')
                    out_name = parts[0].strip()
                    rhs = parts[1].strip()
                    g_type = rhs[:rhs.find('(')].strip().upper()
                    in_str = rhs[rhs.find('(')+1:-1]
                    inputs = [x.strip() for x in in_str.split(',')] if in_str else []
                    
                    self.gate_types[out_name] = g_type
                    if out_name in self.name_to_idx:
                        dst = self.name_to_idx[out_name]
                        for inp in inputs:
                            if inp in self.name_to_idx:
                                src = self.name_to_idx[inp]
                                self.edges_list.append([src, dst])
                                self.gate_inputs[dst].append(src)

        self.adj = {i: [] for i in range(self.num_nodes)}
        self.parents = {i: [] for i in range(self.num_nodes)}
        for src, dst in self.edges_list:
            self.adj[src].append(dst)
            self.adj[dst].append(src)
            self.parents[dst].append(src)

        self.edge_index = torch.tensor(self.edges_list, dtype=torch.long).t().contiguous()
        if self.edge_index.numel() == 0: self.edge_index = torch.empty((2, 0), dtype=torch.long)
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