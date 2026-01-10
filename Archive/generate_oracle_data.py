import os
import torch
import random
import time
from pysat.solvers import Glucose3
from pysat.formula import CNF
from tqdm import tqdm

from WireFaultMiter import WireFaultMiter
from neuro_utils import FastGraphExtractor

# --- CONFIG ---
BENCH_DIR = "../synthetic_bench" # Point to your synthetic files
DATASET_PATH = "dataset_oracle.pt"
SAMPLES_PER_FILE = 50

def generate_oracle_dataset():
    print(f"--- MINING ORACLE DATA (Inverse SAT Solutions) ---")
    dataset = []
    
    if not os.path.exists(BENCH_DIR):
        print(f"Error: {BENCH_DIR} not found.")
        return

    files = [f for f in os.listdir(BENCH_DIR) if f.endswith('.bench')]
    
    for filename in tqdm(files, desc="Mining Circuits"):
        filepath = os.path.join(BENCH_DIR, filename)
        try:
            miter = WireFaultMiter(filepath)
            if not miter.gates: continue
            
            extractor = FastGraphExtractor(filepath, miter.var_map)
            input_set = set(miter.inputs)
            
            for _ in range(SAMPLES_PER_FILE):
                target_gate = random.choice(miter.gates)[0]
                
                # 1. Ask Teacher (SAT Solver) for the Answer
                clauses = miter.build_miter(target_gate, None, 1)
                cnf = CNF(); cnf.extend(clauses)
                
                with Glucose3(bootstrap_with=cnf) as solver:
                    if solver.solve():
                        model = solver.get_model()
                        if not model: continue
                        
                        # 2. Extract The Input Settings
                        model_map = {abs(m): (1.0 if m > 0 else 0.0) for m in model}
                        
                        # 3. Create Graph Data
                        data = extractor.get_data_for_fault(target_gate)
                        
                        y_target = torch.zeros(len(data.node_names), 1)
                        mask = torch.zeros(len(data.node_names), 1)
                        
                        for i, name in enumerate(data.node_names):
                            if name in input_set: 
                                vid = miter.var_map.get(name)
                                if vid in model_map:
                                    y_target[i] = model_map[vid]
                                    mask[i] = 1.0 # Mask=1 means "Learn this Input"
                        
                        data.y = y_target
                        data.train_mask = mask
                        dataset.append(data)

        except Exception as e:
            pass 

    print(f"--- Mining Complete. Collected {len(dataset)} proven solutions. ---")
    torch.save(dataset, DATASET_PATH)

if __name__ == "__main__":
    generate_oracle_dataset()