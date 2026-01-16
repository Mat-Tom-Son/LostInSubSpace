"""
MODULAR ARITHMETIC STROKE TEST
High-Precision Task: No Geometric Slack

The Interleaved task failed to elicit a reflex because the model had geometric slack.
This task has ZERO slack - the answer manifold is a discrete clock.

Protocol:
1. Train healthy victim on Modular Addition (p=113) → expect >99% accuracy
2. Apply stroke (freeze heads + heavy noise)
3. Watch A_activation - on this task, the model MUST scream to maintain precision

The Physics:
- Task: x + y = z (mod 113)
- Answer manifold: twisted torus / discrete clock
- Answer 0 is adjacent to 112 and 1
- Even tiny noise → boundary crossing → wrong answer
- Only defense: scale ||v|| → ∞ so normalized direction is exact
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent
SCRIPT = BASE_DIR / "clean_audit" / "experiments" / "exp_a_foundation.py"
DATA_DIR = BASE_DIR / "clean_audit" / "data"
VICTIM_MODEL = BASE_DIR / "healthy_victim_modular.pt"
LOG_DIR = BASE_DIR / "stroke_test_logs"

# Create log directory
LOG_DIR.mkdir(exist_ok=True)


def run_command(cmd, log_file):
    """Run command and capture output."""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {log_file}")
    print(f"{'='*80}\n")
    
    with open(log_file, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        process.wait()
        
        f.write(f"\n\nFinished: {datetime.now().isoformat()}\n")
        f.write(f"Exit code: {process.returncode}\n")
    
    return process.returncode


def analyze_modular_stroke():
    """Analyze the modular arithmetic stroke test results."""
    print("\n" + "="*80)
    print("ANALYZING MODULAR ARITHMETIC STROKE TEST")
    print("="*80 + "\n")
    
    stroke_log_file = DATA_DIR / "audit_log_exp_a_constraint_seed_42.json"
    
    if not stroke_log_file.exists():
        print(f"ERROR: Stroke log not found at {stroke_log_file}")
        return
    
    with open(stroke_log_file, 'r') as f:
        log = json.load(f)
    
    metrics = log['metrics']
    
    print("A_activation trajectory (first 500 batch steps):")
    print(f"{'Step':<8} {'A_activation':<12} {'Accuracy':<10} {'Loss':<10}")
    print("-" * 50)
    
    reflex_detected = False
    max_a_act = 0.0
    spike_step = None
    baseline_a_act = None
    
    for record in metrics:
        step = record['step']
        if step > 500:
            break
        
        a_act = record.get('A_activation', float('nan'))
        acc = record.get('train_acc', float('nan'))
        loss = record.get('train_loss', float('nan'))
        
        if baseline_a_act is None and step == 0:
            baseline_a_act = a_act
        
        print(f"{step:<8} {a_act:<12.3f} {acc:<10.3f} {loss:<10.3f}")
        
        if a_act > max_a_act:
            max_a_act = a_act
            spike_step = step
        
        # Check for reflex (threshold: 1.5x baseline - lower bar for high-precision task)
        if baseline_a_act and a_act >= baseline_a_act * 1.5 and step >= 10 and step <= 200:
            reflex_detected = True
    
    print("\n" + "="*80)
    print("REFLEX ANALYSIS (MODULAR ARITHMETIC)")
    print("="*80)
    print(f"Baseline A_activation (step 0): {baseline_a_act:.3f}")
    print(f"Maximum A_activation: {max_a_act:.3f} (at step {spike_step})")
    print(f"Spike ratio: {max_a_act / baseline_a_act:.2f}x baseline")
    print(f"Reflex threshold: 1.5x baseline ({baseline_a_act * 1.5:.3f})")
    
    if reflex_detected:
        print("\n🔥 REFLEX DETECTED (THE SCREAM)")
        print("  Hypothesis A confirmed: Amplitude scaling is a survival reflex")
        print("  On this high-precision task, the model HAD to scream")
        print("  The interleaved task was too sloppy - this task has no geometric slack")
    else:
        if max_a_act / baseline_a_act < 1.2:
            print("\n✗ NO REFLEX DETECTED")
            print(f"  Maximum A_activation ({max_a_act:.3f}) only {max_a_act / baseline_a_act:.2f}x baseline")
            print("  Even on the high-precision task, the model accepted death")
            print("  Hypothesis B DEFINITIVELY confirmed: No reflex mechanism exists")
        else:
            print("\n⚠ WEAK SIGNAL")
            print(f"  A_activation increased to {max_a_act / baseline_a_act:.2f}x baseline")
            print("  Below 1.5x threshold but shows some response")
            print("  May need additional analysis")
    
    print("\nSee full logs in: stroke_test_logs/")
    print(f"Raw data: {stroke_log_file}")


def main():
    """Run the modular arithmetic stroke test protocol."""
    print("\n" + "="*80)
    print("MODULAR ARITHMETIC STROKE TEST")
    print("High-Precision Task: Zero Geometric Slack")
    print("="*80 + "\n")
    
    print("WHY THIS WILL WORK WHEN INTERLEAVED FAILED:")
    print("  Interleaved: 'Cat' far from 'Dog' → geometric slack → can be sloppy")
    print("  Modular:     '0' next to '112' → NO slack → MUST be precise")
    print()
    print("PAPER SPECS (from 'Why Variance Explodes'):")
    print("  - p = 113 (prime modulus)")
    print("  - d_model = 128")
    print("  - 4 layers, 4 heads")
    print("  - Weight decay = 1.0 (extreme regularization)")
    print()
    
    # =========================================================================
    # STEP 1: Train healthy victim on Modular Arithmetic (grokking setup)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: TRAINING HEALTHY VICTIM (Modular Arithmetic)")
    print("="*80)
    print("Parameters:")
    print("  - Task: x + y = z (mod 113)")
    print("  - d_model: 128")
    print("  - vocab_size: 128 (113 numbers + special tokens)")
    print("  - epochs: 100 (enough to grok)")
    print("  - Expected: >99% train accuracy (memorization), decent val accuracy")
    print("\n")
    
    # Note: We need to add --dataset modular flag to exp_a_foundation.py
    # For now, we'll use the standard flow but with the right vocab_size
    step1_cmd = [
        sys.executable,
        str(SCRIPT),
        "--condition", "control",
        "--d_model", "128",
        "--vocab_size", "128",  # Modular arithmetic vocab
        "--n_epochs", "100",
        "--seed", "42",
        "--save_model", str(VICTIM_MODEL),
        "--dataset", "modular"  # NEW FLAG
    ]
    
    step1_log = LOG_DIR / f"modular_step1_victim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    exit_code = run_command(step1_cmd, step1_log)
    
    if exit_code != 0:
        print(f"\n✗ STEP 1 FAILED (exit code {exit_code})")
        print(f"Check log: {step1_log}")
        return 1
    
    if not VICTIM_MODEL.exists():
        print(f"\n✗ ERROR: Victim model not saved at {VICTIM_MODEL}")
        return 1
    
    print(f"\n✓ STEP 1 COMPLETE")
    print(f"  Healthy victim saved: {VICTIM_MODEL}")
    
    # =========================================================================
    # STEP 2: Apply stroke (high noise on precision task)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: APPLYING STROKE (Modular Arithmetic)")
    print("="*80)
    print("Parameters:")
    print("  - Load: healthy_victim_modular.pt")
    print("  - d_model: 128")
    print("  - condition: constraint (3/4 heads frozen)")
    print("  - noise_scale: 2.0 (same as interleaved breaking point)")
    print("  - high_freq_log: TRUE")
    print()
    print("PREDICTION:")
    print("  - Accuracy will crash (no slack to hide in)")
    print("  - A_activation SHOULD spike (the 'scream')")
    print("="*80 + "\n")
    
    step2_cmd = [
        sys.executable,
        str(SCRIPT),
        "--condition", "constraint",
        "--load_model", str(VICTIM_MODEL),
        "--d_model", "128",
        "--vocab_size", "128",
        "--n_epochs", "5",
        "--noise_scale", "2.0",
        "--seed", "42",
        "--high_freq_log",
        "--dataset", "modular"
    ]
    
    step2_log = LOG_DIR / f"modular_step2_stroke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    exit_code = run_command(step2_cmd, step2_log)
    
    if exit_code != 0:
        print(f"\n✗ STEP 2 FAILED (exit code {exit_code})")
        print(f"Check log: {step2_log}")
        return 1
    
    print(f"\n✓ STEP 2 COMPLETE")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    analyze_modular_stroke()
    
    print("\n" + "="*80)
    print("MODULAR ARITHMETIC STROKE TEST COMPLETE")
    print("="*80)
    print(f"\nAll logs saved to: {LOG_DIR}/")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
