"""
STROKE TEST ORCHESTRATION SCRIPT - CORRECTED VERSION
Uses d=64 and noise=0.3 to create a REAL stroke effect.

The problem with d=40:
- Victim was too weak (76% accuracy, not 95%)
- Stroke was too gentle (75% accuracy, only -1% drop)
- No crisis = No reflex

The fix:
- d=64: Enough capacity for genuine high performance (>95%)
- 50 epochs: Let victim become confident and "addicted" to geometry
- noise_scale=0.3: Violence - drop from 95% → 40%
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
VICTIM_MODEL = BASE_DIR / "healthy_victim_d64.pt"
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

def analyze_stroke_response():
    """Analyze the stroke test results."""
    print("\n" + "="*80)
    print("ANALYZING STROKE TEST RESULTS")
    print("="*80 + "\n")

    # Load the stroke test log
    stroke_log_file = DATA_DIR / "audit_log_exp_a_constraint_seed_42.json"

    if not stroke_log_file.exists():
        print(f"ERROR: Stroke log not found at {stroke_log_file}")
        return

    with open(stroke_log_file, 'r') as f:
        log = json.load(f)

    metrics = log['metrics']

    # Extract A_activation trajectory for first 500 steps
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

        # Baseline is first step (loaded model)
        if baseline_a_act is None and step == 0:
            baseline_a_act = a_act

        print(f"{step:<8} {a_act:<12.3f} {acc:<10.3f} {loss:<10.3f}")

        if a_act > max_a_act:
            max_a_act = a_act
            spike_step = step

        # Check for reflex (threshold: 2x baseline)
        if baseline_a_act and a_act >= baseline_a_act * 2.0 and step >= 50 and step <= 200:
            reflex_detected = True

    print("\n" + "="*80)
    print("REFLEX ANALYSIS")
    print("="*80)
    print(f"Baseline A_activation (step 0): {baseline_a_act:.3f}")
    print(f"Maximum A_activation: {max_a_act:.3f} (at step {spike_step})")
    print(f"Spike ratio: {max_a_act / baseline_a_act:.2f}x baseline")
    print(f"Reflex threshold: 2.0x baseline ({baseline_a_act * 2:.3f})")

    if reflex_detected:
        print("\n✓ REFLEX DETECTED")
        print("  Hypothesis A confirmed: Amplitude scaling is a survival reflex")
        print("  The model 'screamed' immediately when geometric routes were blocked")
    else:
        if max_a_act / baseline_a_act < 1.5:
            print("\n✗ NO REFLEX DETECTED")
            print(f"  Maximum A_activation ({max_a_act:.3f}) only {max_a_act / baseline_a_act:.2f}x baseline")
            print("  Hypothesis B: Amplitude is purely learned, not reflexive")
        else:
            print("\n⚠ WEAK SIGNAL")
            print(f"  A_activation increased to {max_a_act / baseline_a_act:.2f}x baseline")
            print("  Below 2.0x threshold but above noise floor")
            print("  May need stronger constraints or different architecture")

    print("\nSee full logs in: stroke_test_logs/")
    print(f"Raw data: {stroke_log_file}")

def main():
    """Run the complete stroke test protocol."""
    print("\n" + "="*80)
    print("STROKE TEST PROTOCOL - CORRECTED VERSION")
    print("Testing: Is amplitude scaling a reflex or learned compensation?")
    print("="*80 + "\n")

    print("CORRECTIONS FROM PREVIOUS ATTEMPT:")
    print("  ✗ Old: d=40 (too constrained, victim reached only 76%)")
    print("  ✓ New: d=64 (enough capacity for >95% performance)")
    print()
    print("  ✗ Old: noise_scale=0.15 (too weak, only -1% accuracy drop)")
    print("  ✓ New: noise_scale=0.3 (violent, expect 95% → 40% drop)")
    print()
    print("  ✗ Old: 20 epochs (victim not confident)")
    print("  ✓ New: 50 epochs (victim 'addicted' to geometric solution)")
    print()

    # =========================================================================
    # STEP 1: Train the "Healthy Victim" (d=64, control, 50 epochs)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: TRAINING HEALTHY VICTIM MODEL")
    print("="*80)
    print("Parameters:")
    print("  - d_model: 64 (n/d = 64x superposition, but manageable)")
    print("  - vocab_size: 4096")
    print("  - condition: control (no constraints)")
    print("  - epochs: 50 (build confidence)")
    print("  - Expected: >95% accuracy, A_activation ~6-7")
    print("\n")

    step1_cmd = [
        sys.executable,
        str(SCRIPT),
        "--condition", "control",
        "--d_model", "64",
        "--vocab_size", "4096",
        "--n_epochs", "50",
        "--seed", "42",
        "--save_model", str(VICTIM_MODEL)
    ]

    step1_log = LOG_DIR / f"step1_training_victim_d64_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    # STEP 2: Apply the "Stroke" (load + constraint + VIOLENCE)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: APPLYING STROKE (Acute Injury - HIGH VIOLENCE)")
    print("="*80)
    print("Parameters:")
    print("  - Load: healthy_victim_d64.pt")
    print("  - d_model: 64")
    print("  - condition: constraint (3/4 heads frozen)")
    print("  - noise_scale: 0.3 (VIOLENT - expect 95% → 40% drop)")
    print("  - epochs: 5")
    print("  - high_freq_log: TRUE (every 10 steps, first 500)")
    print("\n")
    print("EXPECTED TRAJECTORY:")
    print("  Step 0-50:   Accuracy crashes from ~95% → ~40%")
    print("  Step 50-100: A_activation spikes from ~6 → 12+ (THE SCREAM)")
    print("  Step 100+:   Model attempts recovery via amplitude")
    print("="*80 + "\n")

    step2_cmd = [
        sys.executable,
        str(SCRIPT),
        "--condition", "constraint",
        "--load_model", str(VICTIM_MODEL),
        "--d_model", "64",
        "--vocab_size", "4096",
        "--n_epochs", "5",
        "--noise_scale", "0.3",
        "--seed", "42",
        "--high_freq_log"
    ]

    step2_log = LOG_DIR / f"step2_stroke_test_d64_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    exit_code = run_command(step2_cmd, step2_log)

    if exit_code != 0:
        print(f"\n✗ STEP 2 FAILED (exit code {exit_code})")
        print(f"Check log: {step2_log}")
        return 1

    print(f"\n✓ STEP 2 COMPLETE")

    # =========================================================================
    # ANALYSIS: Did we observe the reflex?
    # =========================================================================
    analyze_stroke_response()

    print("\n" + "="*80)
    print("STROKE TEST COMPLETE")
    print("="*80)
    print(f"\nAll logs saved to: {LOG_DIR}/")
    print("See STROKE_TEST.md for interpretation guidelines")

    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
