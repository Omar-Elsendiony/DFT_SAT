#!/usr/bin/env python3
"""
Test what happens when Yosys processes a file with escaped identifiers
"""

import subprocess
import tempfile
import os

# Sample from the problematic file
test_verilog = """
module top ( 
    \\a[0] , \\a[1] , \\b[0] , \\b[1] , \\f[0] , \\f[1]  );
  input  \\a[0] , \\a[1] , \\b[0] , \\b[1] ;
  output \\f[0] , \\f[1] ;
  wire n1, n2, n3, n4;
  
  assign n1 = \\a[0]  & ~\\b[0] ;
  assign n2 = ~\\a[0]  & \\b[0] ;
  assign \\f[0]  = n1 | n2;
  assign n3 = \\a[0]  & \\b[0] ;
  assign n4 = ~\\a[1]  & ~\\b[1] ;
  assign \\f[1]  = n3 | n4;
endmodule
"""

print("=" * 80)
print("TESTING YOSYS WITH ESCAPED IDENTIFIERS")
print("=" * 80)

with tempfile.TemporaryDirectory() as tmpdir:
    # Write test file
    input_file = os.path.join(tmpdir, "test.v")
    with open(input_file, 'w') as f:
        f.write(test_verilog)
    
    output_file = os.path.join(tmpdir, "output.v")
    
    # Create synthesis script
    script = f"""
read_verilog {input_file}
hierarchy -check -auto-top
proc; opt; fsm; opt; memory; opt
flatten
techmap; opt
abc -g AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT
techmap; opt
opt_clean -purge
write_verilog -noattr -noexpr {output_file}
"""
    
    script_file = os.path.join(tmpdir, "script.ys")
    with open(script_file, 'w') as f:
        f.write(script)
    
    # Run Yosys
    print("\n🔧 Running Yosys synthesis...")
    print("-" * 80)
    
    result = subprocess.run(
        ['yosys', '-s', script_file],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    print("\n📊 RESULTS:")
    print("-" * 80)
    print(f"Return code: {result.returncode}")
    
    if result.returncode != 0:
        print("\n❌ YOSYS FAILED!")
        print("\n🔴 STDERR:")
        print(result.stderr)
        print("\n📝 STDOUT:")
        print(result.stdout)
    else:
        print("\n✅ YOSYS SUCCEEDED!")
        
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                content = f.read()
            
            print(f"\n📄 Output file created ({len(content)} bytes)")
            print("\nFirst 50 lines:")
            print("-" * 80)
            for i, line in enumerate(content.split('\n')[:50], 1):
                print(f"{i:3d}: {line}")
        else:
            print("\n❌ Output file NOT created!")

print("\n" + "=" * 80)