import json
import numpy as np
import math

def ci(data):
    arr = np.array(data)
    mean = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(len(arr))
    return mean, 1.96 * se

# Exp 1A: Stroke
try:
    with open('clean_audit/data/exp_1a_stroke_results.json') as f:
        res = json.load(f)
    print("TABLE 1 (Stroke):")
    sigmas = [0.0, 0.3, 2.0, 3.0]
    for s in sigmas:
        accs = [run[str(s)]['acc'] for run in res]
        margins = [run[str(s)]['margin'] for run in res]
        m_a, c_a = ci(accs)
        m_m, c_m = ci(margins)
        print(f"Sigma {s}: Acc={m_a:.1%} ± {c_a:.1%} | Margin={m_m:.2f} ± {c_m:.2f}")
except Exception as e:
    print(f"Exp 1A Error: {e}")

# Exp 3: Orthogonal
try:
    with open('clean_audit/data/exp_3_multiseed_results.json') as f:
        res = json.load(f)
    print("\nTABLE 6 (Orthogonal):")
    anc = [r['anchor']['final_acc'] for r in res]
    prb = [r['probe']['final_acc'] for r in res]
    cos = [r['analysis']['final_cosim'] for r in res]
    m_a, c_a = ci(anc)
    m_p, c_p = ci(prb)
    m_c, c_c = ci(cos)
    print(f"Anchor: {m_a:.3%} ± {c_a:.3%}")
    print(f"Probe:  {m_p:.3%} ± {c_p:.3%}")
    print(f"CosSim: {m_c:.4f} ± {c_c:.4f}")
except Exception as e:
    print(f"Exp 3 Error: {e}")
