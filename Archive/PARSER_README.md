# Bench Parser Refactoring

## Overview

This refactoring creates a **shared parsing module** (`BenchParser.py`) for `.bench` files that is used by both the SAT-based fault analysis (`WireFaultMiter`) and the GNN-based neural oracle (`FastGraphExtractor`).

## Key Features

### 1. **Full-Scan DFF Handling**

The parser treats all D Flip-Flops (DFFs) as **full-scan** elements:

- **DFF Q output** → Treated as **Pseudo Primary Input (PPI)**
- **DFF D input** → Treated as **Pseudo Primary Output (PPO)**
- The circuit is "broken" at flip-flops to eliminate feedback loops

This approach is standard in DFT (Design for Testability) analysis and simplifies the combinational logic analysis.

**Example:**
```
G10 = DFF(G29)
```
- `G10` (Q output) becomes a PPI
- `G29` (D input) becomes a PPO
- No gate structure is created for the DFF itself

### 2. **Back Edge Tracking**

The parser maintains a `back_edges` dictionary that maps each wire to all gates it drives. This enables efficient **reverse traversal** of the circuit graph.

**Structure:**
```python
self.back_edges = {
    'G0': ['G28', 'G31', 'G35'],  # G0 drives these gates
    'G1': ['G32', 'G33', 'G36'],
    ...
}
```

**Use Cases:**
- Finding all fanout gates of a wire
- Backward propagation algorithms
- Fault effect analysis
- Graph construction for GNN models

### 3. **Shared Parsing Logic**

Both `WireFaultMiter` and `FastGraphExtractor` now use the same `BenchParser` class, eliminating code duplication and ensuring consistency.

## File Structure

```
DFT_SAT/
├── BenchParser.py          # NEW: Shared parsing module
├── WireFaultMiter.py       # UPDATED: Now uses BenchParser
├── neuro_utils.py          # UPDATED: FastGraphExtractor uses BenchParser
├── benchmark_oracle.py     # UPDATED: Removed _parse_bench() calls
├── generate_oracle_data.py # UPDATED: Removed _parse_bench() calls
├── train_oracle.py         # No changes needed
└── test_parser.py          # NEW: Test script
```

## BenchParser API

### Initialization
```python
from BenchParser import BenchParser

parser = BenchParser("path/to/circuit.bench")
```

### Main Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `inputs` | `list[str]` | Primary Inputs (PIs) |
| `outputs` | `list[str]` | Primary Outputs (POs) |
| `ppis` | `list[str]` | Pseudo Primary Inputs (DFF Q outputs) |
| `ppos` | `list[str]` | Pseudo Primary Outputs (DFF D inputs) |
| `all_inputs` | `list[str]` | PIs + PPIs combined |
| `all_outputs` | `list[str]` | POs + PPOs combined |
| `gates` | `list[tuple]` | List of (output, gate_type, inputs) |
| `gate_dict` | `dict` | Map: output → (gate_type, inputs) |
| `dffs` | `list[tuple]` | List of (Q_output, D_input) pairs |
| `dff_map` | `dict` | Map: Q_output → D_input |
| `back_edges` | `dict` | Map: wire → [gates_it_drives] |
| `var_map` | `dict` | Map: wire_name → SAT variable ID |

### Key Methods

#### `build_var_map()`
Creates a mapping from wire names to SAT solver variable IDs (1-indexed).

```python
var_map = parser.build_var_map()
# Returns: {'G0': 1, 'G1': 2, 'G10': 3, ...}
```

#### `get_fanout(wire_name)`
Returns all gates driven by a wire (forward traversal).

```python
fanout = parser.get_fanout('G0')
# Returns: ['G28', 'G31', 'G35', ...]
```

#### `get_fanin(wire_name)`
Returns the inputs to the gate that produces this wire (backward traversal).

```python
fanin = parser.get_fanin('G28')
# Returns: ['G130']  # If G28 = NOT(G130)
```

#### Type Checking Methods

```python
parser.is_pi('G0')          # True if Primary Input
parser.is_po('G117')        # True if Primary Output
parser.is_ppi('G10')        # True if DFF Q output
parser.is_ppo('G29')        # True if DFF D input
parser.is_dff_output('G10') # True if DFF Q output
```

#### `get_dff_input(q_output)`
Returns the D input for a given DFF Q output.

```python
d_input = parser.get_dff_input('G10')
# Returns: 'G29'
```

#### `get_gate_type(wire_name)`
Returns the gate type that produces this wire.

```python
gate_type = parser.get_gate_type('G28')
# Returns: 'NOT' (or 'INPUT', 'PPI', 'AND', 'OR', etc.)
```

## Migration Guide

### Before (Old Code)
```python
# WireFaultMiter
miter = WireFaultMiter("circuit.bench")
miter._parse_bench()  # Manual parse call
gates = miter.gates

# FastGraphExtractor
var_map = {...}  # Had to be passed in
extractor = FastGraphExtractor("circuit.bench", var_map)
```

### After (New Code)
```python
# WireFaultMiter
miter = WireFaultMiter("circuit.bench")  # Auto-parses
gates = miter.gates

# FastGraphExtractor
extractor = FastGraphExtractor("circuit.bench")  # var_map is optional
# OR
extractor = FastGraphExtractor("circuit.bench", existing_var_map)
```

### Breaking Changes
1. **Removed**: `WireFaultMiter._parse_bench()` method (now called automatically in `__init__`)
2. **Changed**: `FastGraphExtractor.__init__()` no longer requires `var_map` parameter (optional)
3. **Removed**: Inline parsing code in both classes

## Bench File Format

The parser supports the standard `.bench` format as documented at:
https://sportlab.usc.edu/~msabrishami/benchmark-project/bench.html

**Supported gate types:**
- Combinational: `AND`, `OR`, `NOT`, `NAND`, `NOR`, `XOR`, `BUFF`
- Sequential: `DFF` (treated as scan element)

**Example file structure:**
```bench
# Comments start with #
INPUT(G0)
INPUT(G1)
OUTPUT(G117)

G10 = DFF(G29)       # DFF: G10 is PPI, G29 is PPO
G28 = NOT(G130)      # Inverter
G31 = AND(G10, G45, G13)  # 3-input AND gate
```

## Testing

Run the test script to verify the parser works correctly:

```bash
cd DFT_SAT
python test_parser.py
```

This will:
1. Parse a sample ISCAS89 benchmark
2. Display parser statistics
3. Test WireFaultMiter integration
4. Test FastGraphExtractor integration
5. Verify miter and graph construction

## Benefits

1. **Code Reusability**: Single source of truth for parsing logic
2. **Consistency**: SAT and GNN models use identical circuit representations
3. **Maintainability**: Changes to parsing logic only need to be made once
4. **Correctness**: Full-scan DFF handling is now standardized
5. **Performance**: Back edges enable efficient bidirectional traversal
6. **Extensibility**: Easy to add new features (e.g., additional gate types)

## References

- [Bench Format Specification](https://sportlab.usc.edu/~msabrishami/benchmark-project/bench.html)
- ISCAS'89 Sequential Benchmarks: `hdl-benchmarks/iscas89/bench/`
- DFT Scan Chain Theory: Full-scan design breaks sequential loops for testability
