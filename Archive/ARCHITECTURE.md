# Architecture Diagram

## Before Refactoring

```
┌──────────────────────┐         ┌───────────────────────┐
│  WireFaultMiter.py   │         │  neuro_utils.py       │
│                      │         │                       │
│  ┌────────────────┐  │         │  ┌─────────────────┐  │
│  │ _parse_bench() │  │         │  │ Inline Parsing  │  │
│  │  - Parse inputs│  │         │  │  - Parse gates  │  │
│  │  - Parse DFFs  │  │         │  │  - Build edges  │  │
│  │  - Build gates │  │         │  │  - Extract info │  │
│  └────────────────┘  │         │  └─────────────────┘  │
│                      │         │                       │
│  ┌────────────────┐  │         │  ┌─────────────────┐  │
│  │ build_miter()  │  │         │  │ get_data_for_   │  │
│  │                │  │         │  │    _fault()     │  │
│  └────────────────┘  │         │  └─────────────────┘  │
└──────────────────────┘         └───────────────────────┘
         ↓                                   ↓
    SAT Solver                          GNN Model
   (CNF Clauses)                    (Graph Features)
```

**Problems:**
- ❌ Code duplication (~80+ lines of parsing code)
- ❌ Inconsistent DFF handling
- ❌ No back edge tracking
- ❌ Hard to maintain


## After Refactoring

```
                    ┌──────────────────────────┐
                    │   BenchParser.py         │
                    │  (Shared Parsing Core)   │
                    │                          │
                    │  • Parse .bench format   │
                    │  • Handle full-scan DFFs │
                    │  • Track back edges      │
                    │  • Build var_map         │
                    │  • Unified API           │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
         ┌──────────▼───────────┐   ┌─────────▼────────────┐
         │ WireFaultMiter.py    │   │  neuro_utils.py      │
         │                      │   │  FastGraphExtractor  │
         │ self.parser =        │   │                      │
         │   BenchParser(...)   │   │  self.parser =       │
         │                      │   │    BenchParser(...)  │
         │ ┌─────────────────┐  │   │  ┌────────────────┐  │
         │ │ build_miter()   │  │   │  │ get_data_for_  │  │
         │ │  - Uses gates   │  │   │  │   _fault()     │  │
         │ │  - Uses var_map │  │   │  │  - Uses edges  │  │
         │ │  - Uses inputs  │  │   │  │  - Uses types  │  │
         │ └─────────────────┘  │   │  └────────────────┘  │
         └──────────┬───────────┘   └──────────┬───────────┘
                    │                          │
                    ▼                          ▼
              SAT Solver                  GNN Model
             (CNF Clauses)            (Graph Features)
```

**Benefits:**
- ✅ Single source of truth
- ✅ Consistent DFF handling (full-scan)
- ✅ Back edge tracking for bidirectional traversal
- ✅ Easy to maintain and extend


## Data Flow

```
.bench File (s298.bench)
        │
        ▼
┌─────────────────────────────────────────┐
│         BenchParser.__init__()          │
│                                         │
│  1. Parse INPUT/OUTPUT declarations     │
│  2. Parse DFF declarations              │
│     - Q output → PPI (Pseudo PI)        │
│     - D input → PPO (Pseudo PO)         │
│  3. Parse gate declarations             │
│     - Extract: (output, type, inputs)   │
│  4. Build back_edges dictionary         │
│     - Map: wire → gates_it_drives       │
│  5. Build var_map (for SAT)             │
│     - Map: wire → variable_id           │
└─────────────────┬───────────────────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
      ▼                       ▼
┌─────────────┐      ┌────────────────┐
│ SAT Branch  │      │  GNN Branch    │
└─────────────┘      └────────────────┘
      │                       │
      ▼                       ▼
┌──────────────────┐  ┌──────────────────────┐
│ WireFaultMiter   │  │ FastGraphExtractor   │
│                  │  │                      │
│ • gates          │  │ • edges_list         │
│ • var_map        │  │ • gate_types         │
│ • inputs/outputs │  │ • adjacency matrix   │
│ • scan chains    │  │ • node features      │
└────────┬─────────┘  └──────────┬───────────┘
         │                       │
         ▼                       ▼
   ┌─────────────┐        ┌────────────┐
   │ Miter CNF   │        │ PyG Data   │
   │ Clauses     │        │ Object     │
   └─────────────┘        └────────────┘
```


## Full-Scan DFF Handling

```
Original Circuit with DFF:
┌─────┐    ┌─────┐    ┌─────┐
│ G28 ├───►│ G29 ├───►│ DFF ├─┐
└─────┘    └─────┘    │ G10 │ │
                      └─────┘ │
                         ▲    │
                         └────┘
                      (Feedback)

After Full-Scan Breaking:
┌─────┐    ┌─────┐
│ G28 ├───►│ G29 │────► PPO (Pseudo Primary Output)
└─────┘    └─────┘      "Can be observed like a PO"

┌─────┐
│ G10 │────► PPI (Pseudo Primary Input)
└─────┘      "Can be controlled like a PI"

Result: No feedback loop! Purely combinational.
```


## Back Edge Example

```
Forward Edges (gate structure):
    G0 ─────┐
    G1 ─────┤
            ▼
          ┌────┐
          │AND │──► G28
          └────┘

Back Edges (reverse lookup):
    back_edges = {
        'G0': ['G28', 'G31', 'G35'],  ← G0 drives these gates
        'G1': ['G28', 'G32', 'G36'],  ← G1 drives these gates
        'G28': ['G50', 'G51'],        ← G28 drives these gates
    }

Usage:
    fanout = parser.get_fanout('G0')
    # Returns: ['G28', 'G31', 'G35']
    # "What gates does G0 feed into?"
```


## Integration Example

```python
from BenchParser import BenchParser
from WireFaultMiter import WireFaultMiter
from neuro_utils import FastGraphExtractor

# 1. Parse the circuit
bench_file = "s298.bench"

# 2. Create miter for SAT-based analysis
miter = WireFaultMiter(bench_file)
# Auto-parses using BenchParser

# 3. Create graph for GNN-based analysis
extractor = FastGraphExtractor(bench_file)
# Auto-parses using BenchParser
# Shares the same var_map if needed

# 4. Both use identical circuit representation!
assert len(miter.gates) == len(extractor.parser.gates)
assert miter.scan_inputs == extractor.parser.ppis
```


## Testing Architecture

```
test_parser.py
      │
      ├─► Test 1: Direct Parser Usage
      │   ├─ Parse s298.bench
      │   ├─ Verify PIs, POs, PPIs, PPOs
      │   ├─ Check DFF mappings
      │   └─ Verify back edges
      │
      ├─► Test 2: WireFaultMiter Integration
      │   ├─ Initialize with parser
      │   ├─ Check gate count
      │   ├─ Verify var_map
      │   └─ Build sample miter
      │
      └─► Test 3: FastGraphExtractor Integration
          ├─ Initialize with parser
          ├─ Check node/edge count
          ├─ Verify features
          └─ Generate graph data
```
