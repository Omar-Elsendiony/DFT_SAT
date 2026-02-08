import torch
import math
from torch_geometric.data import Data
from BenchParser import BenchParser

class VectorizedGraphExtractor:
    """
    High-Performance SCOAP Extractor using Vectorized Tensor Operations.
    Generates 17-dimensional feature vectors (16 Base + 1 Target Value).
    """
    
    TYPE_MAP = {
        'INPUT': 0, 'PPI': 0, 
        'BUFF': 1, 'NOT': 2,
        'AND': 3, 'NAND': 4,
        'OR': 5, 'NOR': 6,
        'XOR': 7, 'XNOR': 7
    }

    def __init__(self, bench_path, var_map=None, device='cpu'):
        self.parser = BenchParser(bench_path)
        self.device = device
        
        if var_map:
            self.var_map = var_map
        else:
            self.var_map = self.parser.build_var_map()
            
        self.ordered_names = sorted(self.var_map.keys(), key=lambda k: self.var_map[k])
        self.name_to_idx = {name: i for i, name in enumerate(self.ordered_names)}
        self.num_nodes = len(self.ordered_names)
        
        # Build structural tensors
        self.edges_list = []
        self.node_types = torch.zeros(self.num_nodes, dtype=torch.long, device=device)
        
        for name, g_type, _ in self.parser.gates:
            if name in self.name_to_idx:
                self.node_types[self.name_to_idx[name]] = self.TYPE_MAP.get(g_type, 1)
        
        for pi in self.parser.inputs:
            if pi in self.name_to_idx: 
                self.node_types[self.name_to_idx[pi]] = self.TYPE_MAP['INPUT']
        for ppi in self.parser.ppis:
            if ppi in self.name_to_idx: 
                self.node_types[self.name_to_idx[ppi]] = self.TYPE_MAP['INPUT']
        
        for out, _, inputs in self.parser.gates:
            if out in self.name_to_idx:
                dst = self.name_to_idx[out]
                for inp in inputs:
                    if inp in self.name_to_idx:
                        src = self.name_to_idx[inp]
                        self.edges_list.append([src, dst])
        
        if self.edges_list:
            self.edge_index = torch.tensor(self.edges_list, dtype=torch.long, device=device).t().contiguous()
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            
        self.masks = {}
        for t_name, t_id in self.TYPE_MAP.items():
            self.masks[t_name] = (self.node_types == t_id)

        self.adj = [[] for _ in range(self.num_nodes)]
        self.parents = [[] for _ in range(self.num_nodes)]
        for src, dst in self.edges_list:
            self.adj[src].append(dst)
            self.parents[dst].append(src)
            
        self.cc0, self.cc1, self.co = self._compute_scoap_vectorized()
        self.x_base = self._build_base_features()

    def _compute_scoap_vectorized(self):
        """Vectorized SCOAP: Forward Controllability & Backward Observability"""
        num_nodes = self.num_nodes
        src_idx, dst_idx = self.edge_index
        
        cc0 = torch.ones(num_nodes, device=self.device)
        cc1 = torch.ones(num_nodes, device=self.device)
        
        mask_and = self.masks['AND'] | self.masks['NAND']
        mask_or  = self.masks['OR'] | self.masks['NOR']
        mask_inv = self.masks['NAND'] | self.masks['NOR'] | self.masks['NOT']
        mask_xor = self.masks['XOR']
        mask_buf_not = self.masks['BUFF'] | self.masks['NOT']
        
        for _ in range(50): 
            cc0_prev, cc1_prev = cc0.clone(), cc1.clone()
            edge_cc0 = cc0[src_idx]
            edge_cc1 = cc1[src_idx]
            
            min_cc0 = torch.zeros(num_nodes, device=self.device).scatter_reduce_(
                0, dst_idx, edge_cc0, reduce='min', include_self=False)
            min_cc1 = torch.zeros(num_nodes, device=self.device).scatter_reduce_(
                0, dst_idx, edge_cc1, reduce='min', include_self=False)
            sum_cc0 = torch.zeros(num_nodes, device=self.device).scatter_add_(0, dst_idx, edge_cc0)
            sum_cc1 = torch.zeros(num_nodes, device=self.device).scatter_add_(0, dst_idx, edge_cc1)
            
            cc0[mask_and] = min_cc0[mask_and] + 1
            cc1[mask_and] = sum_cc1[mask_and] + 1
            cc0[mask_or] = sum_cc0[mask_or] + 1
            cc1[mask_or] = min_cc1[mask_or] + 1
            cc0[mask_buf_not] = min_cc0[mask_buf_not] + 1
            cc1[mask_buf_not] = min_cc1[mask_buf_not] + 1
            cc0[mask_xor] = torch.minimum(sum_cc0[mask_xor], sum_cc1[mask_xor]) + 1
            cc1[mask_xor] = torch.maximum(min_cc0[mask_xor], min_cc1[mask_xor]) + 1
            
            temp_cc0 = cc0.clone()
            cc0[mask_inv] = cc1[mask_inv]
            cc1[mask_inv] = temp_cc0[mask_inv]
            
            mask_input = self.masks['INPUT']
            cc0[mask_input] = 1.0
            cc1[mask_input] = 1.0
            
            if torch.allclose(cc0, cc0_prev) and torch.allclose(cc1, cc1_prev):
                break

        co = torch.full((num_nodes,), 1e6, device=self.device)
        output_indices = [self.name_to_idx[n] for n in self.parser.all_outputs if n in self.name_to_idx]
        if output_indices:
            co[torch.tensor(output_indices, device=self.device)] = 0.0
        
        gate_cc0_sum = torch.zeros(num_nodes, device=self.device).scatter_add_(0, dst_idx, cc0[src_idx])
        gate_cc1_sum = torch.zeros(num_nodes, device=self.device).scatter_add_(0, dst_idx, cc1[src_idx])
        gate_min_sum = torch.zeros(num_nodes, device=self.device).scatter_add_(
            0, dst_idx, torch.minimum(cc0[src_idx], cc1[src_idx]))

        for _ in range(50):
            co_prev = co.clone()
            co_dst = co[dst_idx]
            dst_types = self.node_types[dst_idx]
            side_costs = torch.zeros_like(co_dst)
            
            is_and = (dst_types == self.TYPE_MAP['AND']) | (dst_types == self.TYPE_MAP['NAND'])
            side_costs[is_and] = gate_cc1_sum[dst_idx][is_and] - cc1[src_idx][is_and]
            is_or = (dst_types == self.TYPE_MAP['OR']) | (dst_types == self.TYPE_MAP['NOR'])
            side_costs[is_or] = gate_cc0_sum[dst_idx][is_or] - cc0[src_idx][is_or]
            is_xor = (dst_types == self.TYPE_MAP['XOR'])
            side_costs[is_xor] = gate_min_sum[dst_idx][is_xor] - torch.minimum(cc0[src_idx], cc1[src_idx])[is_xor]
            
            path_costs = co_dst + side_costs + 1
            new_co = torch.zeros_like(co).scatter_reduce_(
                0, src_idx, path_costs, reduce='min', include_self=False)
            co = torch.minimum(co, new_co)
            if torch.allclose(co, co_prev):
                break
        
        return cc0, cc1, co

    def _compute_depth_fast(self, reverse=False):
        """Vectorized Topological Depth"""
        d_vals = torch.zeros(self.num_nodes, device=self.device)
        src_idx, dst_idx = self.edge_index
        prop_src = dst_idx if reverse else src_idx
        prop_dst = src_idx if reverse else dst_idx
        
        for _ in range(50):
            changed = False
            src_depths = d_vals[prop_src]
            new_depths = torch.zeros(self.num_nodes, device=self.device).scatter_reduce_(
                0, prop_dst, src_depths, reduce='amax', include_self=True)
            new_depths = new_depths + 1
            if not torch.allclose(d_vals, new_depths):
                d_vals = new_depths
                changed = True
            if not changed: 
                break
        
        max_d = d_vals.max() if d_vals.max() > 0 else 1.0
        return (d_vals / max_d).unsqueeze(1)

    def _build_base_features(self):
        """Builds 16 Base Features (without target value)"""
        x_type = torch.nn.functional.one_hot(self.node_types, num_classes=8).float()
        fwd_depth = self._compute_depth_fast(reverse=False)
        rev_depth = self._compute_depth_fast(reverse=True)
        
        f_cc0 = torch.log(self.cc0 + 1).unsqueeze(1) / 10.0
        f_cc1 = torch.log(self.cc1 + 1).unsqueeze(1) / 10.0
        f_co  = torch.log(self.co + 1).unsqueeze(1) / 10.0
        
        is_output = torch.zeros((self.num_nodes, 1), device=self.device)
        for name in self.parser.all_outputs:
            if name in self.name_to_idx: 
                is_output[self.name_to_idx[name]] = 1.0
        
        zeros = torch.zeros((self.num_nodes, 2), device=self.device)
        
        return torch.cat([x_type, fwd_depth, rev_depth, zeros, f_cc0, f_cc1, f_co, is_output], dim=1)

    def get_data_for_fault(self, fault_name, fault_type=1):
        """
        Generate Data object with 17 features.
        fault_type: 1 for SA1 (Target=0), 0 for SA0 (Target=1).
        """
        x = self.x_base.clone()
        tid = self.name_to_idx.get(fault_name)
        
        # 17th Feature: Target Value
        target_feat = torch.full((self.num_nodes, 1), 0.5, device=self.device)
        
        if tid is not None:
            x[tid, 10] = 1.0  # Fault location marker
            target_feat[tid] = 0.0 if fault_type == 1 else 1.0
            
            # BFS Distance
            dist = torch.full((self.num_nodes,), -1.0, device=self.device)
            dist[tid] = 0.0
            queue = [tid]
            visited = {tid: 0}
            idx = 0
            
            while idx < len(queue):
                u = queue[idx]
                idx += 1
                d = visited[u]
                if d >= 10: 
                    continue
                
                neighbors = self.adj[u] + self.parents[u]
                for v in neighbors:
                    if v not in visited:
                        visited[v] = d + 1
                        dist[v] = d + 1
                        queue.append(v)
            
            mask_visited = (dist != -1)
            if mask_visited.any():
                max_d = dist.max()
                if max_d == 0: 
                    max_d = 1.0
                x[mask_visited, 11] = 1.0 - (dist[mask_visited] / max_d)
        
        x = torch.cat([x, target_feat], dim=1)
        return Data(x=x, edge_index=self.edge_index, node_names=self.ordered_names)