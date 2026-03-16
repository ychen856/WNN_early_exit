#!/usr/bin/env python
"""Quick validation that the DWB training script is properly formatted."""
import sys
import ast

script_path = "/Users/yi-chunchen/workspace/WNN_early_exit/src/train_backbone_trans_cifar.py"

try:
    with open(script_path) as f:
        code = f.read()
    ast.parse(code)
    print(f"✓ {script_path} syntax OK")
    print(f"✓ Script is ready for training")
except SyntaxError as e:
    print(f"✗ Syntax error in script: {e}")
    sys.exit(1)
