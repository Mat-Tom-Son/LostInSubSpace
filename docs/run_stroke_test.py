"""
STROKE TEST ORCHESTRATION SCRIPT
Runs the complete two-step stroke test protocol automatically.
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
VICTIM_MODEL = BASE_DIR / "healthy_victim_d40.pt"
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

    for record in metrics:
        step = record['step']
        if step > 500:
            break

        a_act = record.get('A_activation', float('nan'))
        acc = record.get('train_acc', float('nan'))
        loss = record.get('train_loss', float('nan'))

        print(f"{step:<8} {a_act:<12.3f} {acc:<10.3f} {loss:<10.3f}")

        if a_act > max_a_act:
            max_a_act = a_act
            spike_step = step

        # Check for reflex (threshold: 12.0)
        if a_act >= 12.0 and step >= 50 and step <= 200:
            reflex_detected = True

    print("\n" + "="*80)
    print("REFLEX ANALYSIS")
    print("="*80)
    print(f"Maximum A_activation: {max_a_act:.3f} (at step {spike_step})")
    print(f"Reflex threshold: 12.0")

    if reflex_detected:
        print("\n✓ REFLEX DETECTED")
        print("  Hypothesis A confirmed: Amplitude scaling is a survival reflex")
        print("  The model 'screamed' immediately when geometric routes were blocked")
    else:
        print("\n✗ NO REFLEX DETECTED")
        print(f"  Maximum A_activation ({max_a_act:.3f}) below threshold (12.0)")
        print("  Hypothesis B: Amplitude is purely learned, not reflexive")

    print("\nSee full logs in: stroke_test_logs/")
    print(f"Raw data: {stroke_log_file}")

def main():
    """Run the complete stroke test protocol."""
    print("\n" + "="*80)
    print("STROKE TEST PROTOCOL")
    print("Testing: Is amplitude scaling a reflex or learned compensation?")
    print("="*80 + "\n")

    # =========================================================================
    # STEP 1: Train the "Healthy Victim" (d=40, control, 20 epochs)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: TRAINING HEALTHY VICTIM MODEL")
    print("="*80)
    print("Parameters:")
    print("  - d_model: 40")
    print("  - vocab_size: 4096")
    print("  - condition: control (no constraints)")
    print("  - epochs: 20")
    print("  - Expected: ~84% accuracy, A_activation ~6.0")
    print("\n")

    step1_cmd = [
        sys.executable,
        str(SCRIPT),
        "--condition", "control",
        "--d_model", "40",
        "--vocab_size", "4096",
        "--n_epochs", "20",
        "--seed", "42",
        "--save_model", str(VICTIM_MODEL)
    ]

    step1_log = LOG_DIR / f"step1_training_victim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    # STEP 2: Apply the "Stroke" (load + constraint + noise + high-freq logging)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: APPLYING STROKE (Acute Injury)")
    print("="*80)
    print("Parameters:")
    print("  - Load: healthy_victim_d40.pt")
    print("  - d_model: 40")
    print("  - condition: constraint (3/4 heads frozen)")
    print("  - noise_scale: 0.15 (post-attention jamming)")
    print("  - epochs: 5")
    print("  - high_freq_log: TRUE (every 10 steps, first 500)")
    print("\n")
    print("WATCH FOR: A_activation spike from ~6.0 → 12.0+ in first 200 steps")
    print("="*80 + "\n")

    step2_cmd = [
        sys.executable,
        str(SCRIPT),
        "--condition", "constraint",
        "--load_model", str(VICTIM_MODEL),
        "--d_model", "40",
        "--vocab_size", "4096",
        "--n_epochs", "5",
        "--noise_scale", "0.15",
        "--seed", "42",
        "--high_freq_log"
    ]

    step2_log = LOG_DIR / f"step2_stroke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
