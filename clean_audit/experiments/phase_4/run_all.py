"""
Phase 4 Runner: Execute All 4-Layer Scaling Experiments

This script runs all Phase 4 experiments to validate the G×S decomposition
at 4 layers. Results are saved to clean_audit/data/.

Usage:
    python -m clean_audit.experiments.phase_4.run_all [--quick_test] [--device cuda]

Or from the clean_audit directory:
    python experiments/phase_4/run_all.py [--quick_test] [--device cuda]

Experiments:
    4.1 Routing Swap: Tests G causality (catastrophic failure on QK swap)
    4.2 Young G Probe: Tests subspace structure (orthogonal S allocations)
    4.3 Sedation Test: Tests margin-as-budget (prophylactic amplitude)

Expected runtime (full):
    - Per experiment: ~2-3 hours on GPU (5 seeds × 20k steps)
    - Total: ~6-9 hours

Quick test mode:
    - Per experiment: ~10-15 minutes
    - Total: ~30-45 minutes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from datetime import datetime

from exp_4_1_swap_4layer import run_experiment as run_swap
from exp_4_2_young_g_4layer import run_experiment as run_young_g
from exp_4_3_sedation_4layer import run_experiment as run_sedation


def run_all_experiments(
    device: str = 'cuda',
    n_seeds: int = 5,
    quick_test: bool = False
):
    """Run all Phase 4 experiments."""

    print("\n" + "="*80)
    print("PHASE 4: 4-LAYER SCALING VALIDATION")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Seeds: {n_seeds}")
    print(f"Quick test: {quick_test}\n")

    results = {}

    # =========================================================================
    # Experiment 4.1: Routing Swap
    # =========================================================================

    print("\n" + "#"*80)
    print("# EXPERIMENT 4.1: ROUTING SWAP (4-LAYER)")
    print("#"*80 + "\n")

    swap_results = run_swap(
        n_layers=4,
        n_steps=4000 if quick_test else 20000,
        n_seeds=1 if quick_test else n_seeds,
        device=device
    )

    results['exp_4_1_swap'] = {
        'pass_rate': sum(1 for r in swap_results if r['success']) / len(swap_results),
        'n_seeds': len(swap_results)
    }

    # =========================================================================
    # Experiment 4.2: Young G Probe
    # =========================================================================

    print("\n" + "#"*80)
    print("# EXPERIMENT 4.2: YOUNG G SUBSPACE PROBE (4-LAYER)")
    print("#"*80 + "\n")

    young_g_results = run_young_g(
        n_layers=4,
        warmup_steps=500 if quick_test else 2500,
        anchor_steps=3000 if quick_test else 20000,
        probe_steps=3000 if quick_test else 20000,
        n_seeds=1 if quick_test else n_seeds,
        device=device
    )

    results['exp_4_2_young_g'] = {
        'pass_rate': sum(1 for r in young_g_results if r['analysis']['overall_pass']) / len(young_g_results),
        'n_seeds': len(young_g_results)
    }

    # =========================================================================
    # Experiment 4.3: Sedation Test
    # =========================================================================

    print("\n" + "#"*80)
    print("# EXPERIMENT 4.3: SEDATION TEST (4-LAYER)")
    print("#"*80 + "\n")

    sedation_results = run_sedation(
        n_layers=4,
        n_steps=4000 if quick_test else 20000,
        n_seeds=1 if quick_test else n_seeds,
        device=device
    )

    results['exp_4_3_sedation'] = {
        'pass_rate': sum(1 for r in sedation_results if r['analysis']['overall_pass']) / len(sedation_results),
        'n_seeds': len(sedation_results)
    }

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "="*80)
    print("PHASE 4 SUMMARY: 4-LAYER SCALING VALIDATION")
    print("="*80 + "\n")

    all_pass = True
    for exp_name, exp_results in results.items():
        status = "✓ PASS" if exp_results['pass_rate'] >= 0.8 else "✗ FAIL"
        if exp_results['pass_rate'] < 0.8:
            all_pass = False
        print(f"{exp_name}: {status} ({exp_results['pass_rate']:.0%} pass rate, n={exp_results['n_seeds']})")

    print("\n" + "-"*40)
    if all_pass:
        print("OVERALL: ✓ G×S DECOMPOSITION VALIDATED AT 4 LAYERS")
    else:
        print("OVERALL: ✗ SOME EXPERIMENTS INCONCLUSIVE")
    print("-"*40)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save summary
    summary_path = Path("clean_audit/data/phase_4_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'n_seeds': n_seeds,
        'quick_test': quick_test,
        'results': results,
        'overall_pass': all_pass
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run all Phase 4 (4-layer) scaling experiments"
    )
    parser.add_argument('--device', type=str, default='cuda',
                        help='Compute device (cuda or cpu)')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Number of seeds per experiment')
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test mode (reduced steps/seeds)')

    args = parser.parse_args()

    run_all_experiments(
        device=args.device,
        n_seeds=args.n_seeds,
        quick_test=args.quick_test
    )


if __name__ == '__main__':
    main()
