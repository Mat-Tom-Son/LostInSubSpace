"""
Quick test to verify clamp is actually working.

This will train for 2 epochs with naive clamp and check that:
1. "[CLAMP ENABLED]" message appears
2. "[CLAMP ACTIVE]" message shows actual norm
3. A_act matches target_norm (within tolerance)
"""

import sys
sys.path.append('clean_audit')

import subprocess
import json
from pathlib import Path

print("="*80)
print("CLAMP FIX VERIFICATION TEST")
print("="*80)

# Run experiment with naive clamp only
print("\nRunning Constraint + Naive Clamp (2 epochs)...")
print("This should show:")
print("  1. [CLAMP ENABLED] message")
print("  2. [CLAMP ACTIVE] with actual norm")
print("  3. A_act locked to target_norm\n")

result = subprocess.run(
    [
        'python', 'clean_audit/experiments/exp_a_foundation.py',
        '--condition', 'naive_clamp',
        '--n_epochs', '2',
        '--vocab_size', '4096',
        '--d_model', '128',
        '--quick_test'
    ],
    capture_output=True,
    text=True
)

print(result.stdout)

# Check for success indicators
output = result.stdout

checks = []

# Check 1: Clamp enabled message
if "[CLAMP ENABLED]" in output:
    print("[OK] CLAMP ENABLED message found")
    checks.append(True)
else:
    print("[FAIL] CLAMP ENABLED message NOT found - clamp not being set!")
    checks.append(False)

# Check 2: Clamp active message
if "[CLAMP ACTIVE]" in output:
    print("[OK] CLAMP ACTIVE message found")
    checks.append(True)
else:
    print("[FAIL] CLAMP ACTIVE message NOT found - clamp not being applied!")
    checks.append(False)

# Check 3: Parse actual norm values
import re
target_norm_match = re.search(r'target_norm:\s+([\d.]+)', output)
active_norm_matches = re.findall(r'\[CLAMP ACTIVE\] Actual resid norm after clamp:\s+([\d.]+)', output)

if target_norm_match and active_norm_matches:
    target_norm = float(target_norm_match.group(1))
    active_norm = float(active_norm_matches[0])  # First epoch

    print(f"\nTarget norm: {target_norm:.4f}")
    print(f"Active norm: {active_norm:.4f}")

    diff = abs(active_norm - target_norm)
    tolerance = 0.5  # Allow small difference due to batching

    if diff < tolerance:
        print(f"[OK] Active norm matches target (diff: {diff:.4f} < {tolerance})")
        checks.append(True)
    else:
        print(f"[FAIL] Active norm DOES NOT match target (diff: {diff:.4f} > {tolerance})")
        print("     Clamp function may not be working correctly!")
        checks.append(False)
else:
    print("[FAIL] Could not parse norm values from output")
    checks.append(False)

# Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

if all(checks):
    print("[OK] ALL CHECKS PASSED - Clamp is working correctly!")
    print("\nYou can now run the full experiment with confidence:")
    print("  python clean_audit/experiments/exp_a_foundation.py --seed 42")
else:
    print("[FAIL] SOME CHECKS FAILED - Clamp may not be working")
    print("\nFailed checks:")
    if not checks[0]:
        print("  - Clamp not being set on model")
    if len(checks) > 1 and not checks[1]:
        print("  - Clamp not being applied during forward pass")
    if len(checks) > 2 and not checks[2]:
        print("  - Clamp not achieving target norm")

print("="*80)
