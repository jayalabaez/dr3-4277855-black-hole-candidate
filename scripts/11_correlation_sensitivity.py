#!/usr/bin/env python3
"""
11 — Correlation sensitivity test for the Monte Carlo mass posterior.

The full NSS covariance matrix is unavailable, so script 03 samples
parameters independently.  This script quantifies the impact of
plausible inter-parameter correlations by constructing synthetic
correlated draws and comparing the resulting M2 posterior to the
independent-sampling baseline.

We test three key correlation pairs at rho = 0, 0.3, 0.5, 0.7:
  (a) parallax vs angular semi-major axis (both enter via a = a_ang / plx)
  (b) period vs eccentricity (dynamical correlation in Keplerian fits)

We also expand the ad-hoc [0.95, 1.05] model systematic to wider
brackets: [0.90, 1.10] and [0.85, 1.15].

Outputs:
  results/correlation_sensitivity_results.json
"""

import json, pathlib, numpy as np

# ── Constants ────────────────────────────────────────────────────────
SOURCE_ID   = 4277855016732107520
PLX         = 1.5228       # mas
PLX_ERR     = 0.1549       # mas
PERIOD      = 424.403      # d
PERIOD_ERR  = 1.159        # d
ECC         = 0.3427
ECC_ERR     = 0.0194
M1_BEST     = 1.340        # Msun
M1_SIGMA    = 0.40         # Msun
M2_CAT      = 12.313       # Msun
M_TOTAL_CAT = M1_BEST + M2_CAT
PLX_INFLATION = 1.7

# Angular semi-major axis: from the total mass, we can back out
# a_ang_mas ~ 3.64 mas (from NSS table).  We assume ~10% uncertainty.
A_ANG       = 3.64         # mas (approximate)
A_ANG_ERR   = 0.36         # mas (~10%)

N_MC = 500_000
rng  = np.random.default_rng(42)

BASEDIR = pathlib.Path(__file__).resolve().parent.parent

# ── Helper: correlated draws from bivariate normal ───────────────────
def draw_correlated_pair(mu1, sig1, mu2, sig2, rho, n):
    """Draw n correlated samples from bivariate normal."""
    mean = [mu1, mu2]
    cov = [[sig1**2, rho * sig1 * sig2],
           [rho * sig1 * sig2, sig2**2]]
    return rng.multivariate_normal(mean, cov, n).T

# ── Baseline: independent sampling (reproduce script 03) ────────────
def draw_m2_independent(plx_inf=1.7, sys_half_width=0.05):
    m1 = np.clip(rng.normal(M1_BEST, M1_SIGMA, N_MC), 0.5, 5.0)
    plx = np.clip(rng.normal(PLX, PLX_ERR * plx_inf, N_MC), 0.1, 10.0)
    P_d = np.clip(rng.normal(PERIOD, PERIOD_ERR, N_MC), 100.0, 2000.0)
    sys_fac = rng.uniform(1.0 - sys_half_width, 1.0 + sys_half_width, N_MC)
    m_total = M_TOTAL_CAT * sys_fac * (PLX / plx)**3 * (PERIOD / P_d)**2
    m2 = m_total - m1
    return m2[m2 > 0.5]

# ── Test A: parallax–angular-semi-major-axis correlation ─────────────
# In the full NSS fit, plx and a_ang are correlated because
# the physical semi-major axis a = a_ang / plx enters Kepler III.
# A positive correlation (larger plx → larger a_ang) would mean
# the M_total uncertainty is partially cancelled (since M_tot ∝ a³/plx³).

def draw_m2_plx_aang_correlated(rho_plx_aang, plx_inf=1.7):
    """Draw M2 with correlated plx and a_ang."""
    m1 = np.clip(rng.normal(M1_BEST, M1_SIGMA, N_MC), 0.5, 5.0)
    P_d = np.clip(rng.normal(PERIOD, PERIOD_ERR, N_MC), 100.0, 2000.0)
    sys_fac = rng.uniform(0.95, 1.05, N_MC)

    # Correlated parallax and angular semi-major axis
    plx_draws, aang_draws = draw_correlated_pair(
        PLX, PLX_ERR * plx_inf,
        A_ANG, A_ANG_ERR,
        rho_plx_aang, N_MC
    )
    plx_draws = np.clip(plx_draws, 0.1, 10.0)
    aang_draws = np.clip(aang_draws, 0.1, 20.0)

    # Physical semi-major axis in AU:  a = a_ang(mas) / plx(mas) * 1 AU
    # But we need to compute M_total from a:
    # M_total = 4pi² a³ / (G P²)
    # Relative to catalog:
    # M_total / M_total_cat = (a / a_cat)³ / (P / P_cat)²
    # where a_cat = A_ANG / PLX
    a_ratio = (aang_draws / plx_draws) / (A_ANG / PLX)
    m_total = M_TOTAL_CAT * sys_fac * a_ratio**3 * (PERIOD / P_d)**2
    m2 = m_total - m1
    return m2[m2 > 0.5]

# ── Test B: period–eccentricity correlation ──────────────────────────
def draw_m2_P_ecc_correlated(rho_P_ecc, plx_inf=1.7):
    """Draw M2 with correlated P and e."""
    m1 = np.clip(rng.normal(M1_BEST, M1_SIGMA, N_MC), 0.5, 5.0)
    plx = np.clip(rng.normal(PLX, PLX_ERR * plx_inf, N_MC), 0.1, 10.0)
    sys_fac = rng.uniform(0.95, 1.05, N_MC)

    P_draws, ecc_draws = draw_correlated_pair(
        PERIOD, PERIOD_ERR, ECC, ECC_ERR, rho_P_ecc, N_MC
    )
    P_draws = np.clip(P_draws, 100.0, 2000.0)
    ecc_draws = np.clip(ecc_draws, 0.0, 0.99)

    # Period enters Kepler III directly; eccentricity doesn't change
    # the total mass but affects K1 and orbital dynamics.
    # For M_total: only P matters (not e), but correlated P-e
    # could shift the effective P distribution.
    m_total = M_TOTAL_CAT * sys_fac * (PLX / plx)**3 * (PERIOD / P_draws)**2
    m2 = m_total - m1
    return m2[m2 > 0.5]

# ── Run all tests ────────────────────────────────────────────────────
def summarize(m2, label):
    return {
        'label': label,
        'median': round(float(np.median(m2)), 1),
        'CI68': [round(float(np.percentile(m2, 16)), 1),
                 round(float(np.percentile(m2, 84)), 1)],
        'CI90': [round(float(np.percentile(m2, 5)), 1),
                 round(float(np.percentile(m2, 95)), 1)],
        'P_gt_5': round(float(np.mean(m2 >= 5.0) * 100), 1),
        'P_gt_10': round(float(np.mean(m2 >= 10.0) * 100), 1),
        'n_valid': int(len(m2)),
    }

print('=' * 70)
print('  CORRELATION SENSITIVITY TEST')
print('=' * 70)

results = {'tests': []}

# Baseline (independent)
rng = np.random.default_rng(42)
m2_base = draw_m2_independent(1.7, 0.05)
s = summarize(m2_base, 'Baseline (independent, sys=5%)')
results['tests'].append(s)
print(f"\n{s['label']}:")
print(f"  Median={s['median']}, 68%CI={s['CI68']}, P(>5)={s['P_gt_5']}%, P(>10)={s['P_gt_10']}%")

# Test A: parallax-aang correlation
for rho in [0.0, 0.3, 0.5, 0.7]:
    rng = np.random.default_rng(42)
    m2 = draw_m2_plx_aang_correlated(rho, 1.7)
    s = summarize(m2, f'plx-aang rho={rho}')
    results['tests'].append(s)
    print(f"\n{s['label']}:")
    print(f"  Median={s['median']}, 68%CI={s['CI68']}, P(>5)={s['P_gt_5']}%, P(>10)={s['P_gt_10']}%")

# Test B: period-eccentricity correlation
for rho in [0.0, 0.3, 0.5, 0.7]:
    rng = np.random.default_rng(42)
    m2 = draw_m2_P_ecc_correlated(rho, 1.7)
    s = summarize(m2, f'P-ecc rho={rho}')
    results['tests'].append(s)
    print(f"\n{s['label']}:")
    print(f"  Median={s['median']}, 68%CI={s['CI68']}, P(>5)={s['P_gt_5']}%, P(>10)={s['P_gt_10']}%")

# Test C: wider systematic brackets
for hw in [0.05, 0.10, 0.15]:
    rng = np.random.default_rng(42)
    m2 = draw_m2_independent(1.7, hw)
    pct = int(hw * 100)
    s = summarize(m2, f'sys={pct}%')
    results['tests'].append(s)
    print(f"\n{s['label']}:")
    print(f"  Median={s['median']}, 68%CI={s['CI68']}, P(>5)={s['P_gt_5']}%, P(>10)={s['P_gt_10']}%")

# Summary table
print('\n' + '=' * 70)
print(f"{'Test':<35s} {'Med':>5s} {'68% CI':>16s} {'P(>5)':>7s} {'P(>10)':>7s}")
print('-' * 70)
for t in results['tests']:
    ci = f"[{t['CI68'][0]}, {t['CI68'][1]}]"
    print(f"{t['label']:<35s} {t['median']:>5.1f} {ci:>16s} {t['P_gt_5']:>6.1f}% {t['P_gt_10']:>6.1f}%")

# Save
outdir = BASEDIR / 'results'
outdir.mkdir(exist_ok=True)
outf = outdir / 'correlation_sensitivity_results.json'
with open(outf, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outf}")
