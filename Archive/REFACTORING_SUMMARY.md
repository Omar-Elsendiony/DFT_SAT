# Bench Parser Refactoring - Summary

## What Changed

### ✅ New Files Created

1. **`BenchParser.py`** - Shared parsing module for .bench files
   - Handles full-scan DFF assumptions
   - Tracks back edges for bidirectional traversal
   - Provides unified API for both SAT and GNN models

2. **`test_parser.py`** - Comprehensive test suite
   - Validates parser functionality
   - Tests integration with WireFaultMiter
   - Tests integration with FastGraphExtractor

3. **`PARSER_README.md`** - Complete documentation
   - API reference
   - Migration guide
   - Usage examples

### 🔄 Files Modified

1. **`WireFaultMiter.py`**
   - Removed inline parsing code
   - Now uses `BenchParser` in `__init__`
   - Removed `_parse_bench()` method
   - Inherits all parsing logic from shared module

2. **`neuro_utils.py`**
   - Updated `FastGraphExtractor` to use `BenchParser`
   - Made `var_map` parameter optional
   - Removed inline parsing code
   - Better handling of PPIs and PIs

3. **`benchmark_oracle.py`**
   - Removed explicit `miter._parse_bench()` calls
   - Parsing now happens automatically in `__init__`

4. **`generate_oracle_data.py`**
   - Removed explicit `miter._parse_bench()` calls
   - Parsing now happens automatically in `__init__`

## Key Features Implemented

### 1. Full-Scan DFF Handling ✅

```python
# Example: G10 = DFF(G29)
parser.ppis     # ['G10', ...]  - DFF Q outputs (Pseudo Primary Inputs)
parser.ppos     # ['G29', ...]  - DFF D inputs (Pseudo Primary Outputs)
parser.all_inputs   # PIs + PPIs
parser.all_outputs  # POs + PPOs
```

**Why?** This breaks the circuit at flip-flops, eliminating feedback loops and enabling combinational analysis.

### 2. Back Edge Dictionary ✅

```python
parser.back_edges = {
    'G0': ['G28', 'G31', 'G35'],  # G0 drives these gates
    'G1': ['G32', 'G33'],
    ...
}

# API method
fanout = parser.get_fanout('G0')  # Returns ['G28', 'G31', 'G35']
```

**Why?** Enables efficient reverse traversal for:
- Fanout analysis
- Backward fault propagation
- Graph construction for neural networks

### 3. Shared Parsing Logic ✅

Both `WireFaultMiter` and `FastGraphExtractor` use the same `BenchParser`:

```python
# Before (duplicated code)
class WireFaultMiter:
    def _parse_bench(self): # 50+ lines of parsing code
        ...

class FastGraphExtractor:
    def __init__(self, ...): # 30+ lines of parsing code
        ...

# After (shared parser)
from BenchParser import BenchParser

class WireFaultMiter:
    def __init__(self, bench_file):
        self.parser = BenchParser(bench_file)  # Done!

class FastGraphExtractor:
    def __init__(self, bench_path):
        self.parser = BenchParser(bench_path)  # Done!
```

## Test Results

```
Testing BenchParser with s298.bench
=====================================
✅ PIs: 3, POs: 6
✅ PPIs: 14, PPOs: 14  (Full-scan DFFs)
✅ Gates: 119, DFFs: 14
✅ Back edges working (G0 → ['I229'])

WireFaultMiter Integration
===========================
✅ Inputs: 17 (3 PIs + 14 PPIs)
✅ Outputs: 20 (6 POs + 14 PPOs)
✅ Gates: 119
✅ Variable Map: 136 nodes
✅ Miter built: 827 CNF clauses

FastGraphExtractor Integration
================================
✅ Nodes: 136
✅ Edges: 244
✅ Features: [136, 14] tensor
✅ Graph data generated successfully
```

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Code Duplication** | 2 parsing implementations | 1 shared parser |
| **DFF Handling** | Inconsistent | Standardized full-scan |
| **Back Edges** | Not tracked | Tracked in `back_edges` dict |
| **Maintainability** | Change in 2 places | Change in 1 place |
| **Testing** | No dedicated tests | Comprehensive test suite |
| **Documentation** | Scattered comments | Detailed README |

## API Highlights

### Core Attributes
```python
parser = BenchParser("circuit.bench")

# Inputs/Outputs
parser.inputs        # ['G0', 'G1', 'G2']
parser.outputs       # ['G117', 'G132', ...]
parser.ppis          # ['G10', 'G11', ...]  (DFF Q)
parser.ppos          # ['G29', 'G30', ...]  (DFF D)
parser.all_inputs    # PIs + PPIs
parser.all_outputs   # POs + PPOs

# Circuit structure
parser.gates         # [(out, type, inputs), ...]
parser.gate_dict     # {out: (type, inputs), ...}
parser.dffs          # [(Q, D), ...]
parser.back_edges    # {wire: [gates_driven], ...}
parser.var_map       # {wire: sat_var_id, ...}
```

### Useful Methods
```python
# Build SAT variable mapping
var_map = parser.build_var_map()

# Fanout analysis
gates_driven = parser.get_fanout('G0')

# Fanin analysis
gate_inputs = parser.get_fanin('G28')

# Type checking
parser.is_pi('G0')           # True
parser.is_ppi('G10')         # True (DFF Q)
parser.is_ppo('G29')         # True (DFF D)
parser.is_dff_output('G10')  # True

# DFF relationship
d_input = parser.get_dff_input('G10')  # Returns 'G29'

# Gate type
gate_type = parser.get_gate_type('G28')  # Returns 'NOT'
```

## Migration Notes

### Breaking Changes
❌ **Removed**: `WireFaultMiter._parse_bench()` - No longer needed
❌ **Changed**: `FastGraphExtractor(path, var_map)` → `FastGraphExtractor(path, var_map=None)`

### Simple Migration
```python
# Old code
miter = WireFaultMiter("circuit.bench")
miter._parse_bench()  # ❌ Remove this line

# New code
miter = WireFaultMiter("circuit.bench")  # ✅ Parsing happens automatically
```

## Files Overview

```
DFT_SAT/
├── BenchParser.py          ← NEW: Shared parser (183 lines)
├── WireFaultMiter.py       ← UPDATED: Uses BenchParser
├── neuro_utils.py          ← UPDATED: Uses BenchParser
├── benchmark_oracle.py     ← UPDATED: Removed _parse_bench() call
├── generate_oracle_data.py ← UPDATED: Removed _parse_bench() call
├── test_parser.py          ← NEW: Test suite
└── PARSER_README.md        ← NEW: Full documentation
```

## Next Steps

1. ✅ Run tests: `python test_parser.py`
2. ✅ Review API documentation: See `PARSER_README.md`
3. ✅ Update any external code that used old parsing methods
4. ✅ Consider extending parser for additional gate types if needed

## Questions?

See `PARSER_README.md` for:
- Complete API reference
- Bench format specification
- Usage examples
- Additional method documentation
