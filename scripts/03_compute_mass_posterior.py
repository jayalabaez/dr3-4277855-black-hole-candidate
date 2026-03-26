#!/usr/bin/env python3
"""
03 — Monte Carlo mass posterior for Gaia DR3 4277855016732107520.

Properly propagates parallax uncertainty through Kepler III.
Also runs a parallax-inflated (×1.7) scenario and a parallax-only stress test.

For an Orbital solution the total mass comes from Kepler III:
    M_total = 4 π² a³ / (G P²)
where the physical semi-major axis  a = a_ang / plx.
Since a_ang is derived from the astrometric orbit and is NOT published
separately, we use the catalog M_total and scale:
    M_total ∝ (plx_nom / plx)³ × (P / P_nom)²   (inverted Kepler III scaling)

    M2 = M_total − M1

This means the ±10.2 % parallax uncertainty propagates as ±30.6 %
in M_total — the dominant error source.

Outputs:
  results/mass_posterior_results.json
  paper/figures/fig_mass_posterior.pdf
"""

import json, pathlib, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────
SOURCE_ID   = 4277855016732107520
PLX         = 1.5228       # mas
PLX_ERR     = 0.1549       # mas
PERIOD      = 424.403      # d
PERIOD_ERR  = 1.159        # d
M1_BEST     = 1.340        # M☉
M1_SIGMA    = 0.40         # M☉  (30% fractional)
M2_CAT      = 12.313       # M☉  (catalogue)
M_TOTAL_CAT = M1_BEST + M2_CAT     # 13.653 M☉
PLX_INFLATION = 1.7        # recommended for Orbital at G ≈ 11

N_MC = 500_000
rng  = np.random.default_rng(42)

BASEDIR = pathlib.Path(__file__).resolve().parent.parent

# ── Helper: draw M2 posterior ────────────────────────────────────────────
def draw_posterior(plx_sigma_factor=1.0, label='nominal'):
    """Return M2 draws with given parallax-error inflation."""
    m1      = rng.normal(M1_BEST, M1_SIGMA, N_MC)
    m1      = np.clip(m1, 0.5, 5.0)

    plx     = rng.normal(PLX, PLX_ERR * plx_sigma_factor, N_MC)
    plx     = np.clip(plx, 0.1, 10.0)

    P_d     = rng.normal(PERIOD, PERIOD_ERR, N_MC)
    P_d     = np.clip(P_d, 100.0, 2000.0)

    # 5 % astrometric-model systematic
    sys_fac = rng.normal(1.0, 0.05, N_MC)

    # Kepler-scaled total mass:
    #   M_total ∝ a³/P²  and  a ∝ 1/plx  →  M_total ∝ plx⁻³ P⁻²
    # Relative to catalogue:
    m_total = M_TOTAL_CAT * sys_fac * (PLX / plx)**3 * (PERIOD / P_d)**2

    m2 = m_total - m1
    valid = m2 > 0.5
    return m2[valid]

# ── Nominal posterior ────────────────────────────────────────────────────
m2_nom = draw_posterior(1.0, 'nominal')

med    = np.median(m2_nom)
lo16   = np.percentile(m2_nom, 16)
hi84   = np.percentile(m2_nom, 84)
lo5    = np.percentile(m2_nom, 5)
hi95   = np.percentile(m2_nom, 95)
P_gt5  = np.mean(m2_nom >= 5.0) * 100
P_gt10 = np.mean(m2_nom >= 10.0) * 100

print('=' * 70)
print('  MASS POSTERIOR  (Gaia DR3 4277855016732107520)')
print('=' * 70)
print(f'  Nominal  (σ_plx × 1.0):')
print(f'    Median M2 = {med:.1f} M☉')
print(f'    68% CI    = [{lo16:.1f}, {hi84:.1f}]')
print(f'    90% CI    = [{lo5:.1f}, {hi95:.1f}]')
print(f'    P(>5)     = {P_gt5:.1f}%')
print(f'    P(>10)    = {P_gt10:.1f}%')
print()

# ── Inflated parallax posterior ──────────────────────────────────────────
m2_inf = draw_posterior(PLX_INFLATION, 'inflated')

med_i   = np.median(m2_inf)
lo16_i  = np.percentile(m2_inf, 16)
hi84_i  = np.percentile(m2_inf, 84)
lo5_i   = np.percentile(m2_inf, 5)
hi95_i  = np.percentile(m2_inf, 95)
P_gt5_i = np.mean(m2_inf >= 5.0) * 100
P_gt10_i= np.mean(m2_inf >= 10.0) * 100

print(f'  Inflated (σ_plx × {PLX_INFLATION}):')
print(f'    Median M2 = {med_i:.1f} M☉')
print(f'    68% CI    = [{lo16_i:.1f}, {hi84_i:.1f}]')
print(f'    90% CI    = [{lo5_i:.1f}, {hi95_i:.1f}]')
print(f'    P(>5)     = {P_gt5_i:.1f}%')
print(f'    P(>10)    = {P_gt10_i:.1f}%')
print()

# ── Parallax-only stress test ────────────────────────────────────────────
def m2_at_plx(p):
    mt = M_TOTAL_CAT * (PLX / p)**3
    return mt - M1_BEST

m2_lo = m2_at_plx(PLX + PLX_ERR)
m2_hi = m2_at_plx(PLX - PLX_ERR)
print(f'  Parallax stress test (plx ± 1σ, all else fixed):')
print(f'    plx+1σ → M2 = {m2_lo:.1f} M☉')
print(f'    plx-1σ → M2 = {m2_hi:.1f} M☉')
print()

# ── Figure: 3-panel ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: M1 sensitivity
m1_vals = np.array([0.8, 1.0, 1.34, 1.8, 2.5])
m2_vals = M_TOTAL_CAT - m1_vals
axes[0].plot(m1_vals, m2_vals, 'ko-', lw=2)
axes[0].axhline(5.0, color='r', ls='--', label='BH threshold (5 M$_\\odot$)')
axes[0].set_xlabel('Assumed $M_1$ (M$_\\odot$)')
axes[0].set_ylabel('$M_2$ (M$_\\odot$)')
axes[0].set_title('$M_2$ sensitivity to $M_1$')
axes[0].legend(fontsize=8)

# Panel 2: nominal + inflated posterior overlay
bins = np.linspace(0, 50, 150)
axes[1].hist(m2_nom, bins=bins, density=True, alpha=0.6,
             color='steelblue', label='Nominal')
axes[1].hist(m2_inf, bins=bins, density=True, alpha=0.5,
             color='darkorange', label=f'Inflated ({PLX_INFLATION}' + r'$\times\sigma_\varpi$)')
axes[1].axvline(5.0, color='r', ls='--')
axes[1].set_xlabel('$M_2$ (M$_\\odot$)')
axes[1].set_ylabel('Posterior density')
axes[1].set_title('Companion mass posterior')
axes[1].set_xlim(0, 50)
axes[1].legend(fontsize=8)

# Panel 3: parallax stress test
plx_arr = np.linspace(PLX - 3*PLX_ERR, PLX + 3*PLX_ERR, 200)
m2_arr  = m2_at_plx(plx_arr)
axes[2].plot(plx_arr, m2_arr, 'k-', lw=2)
axes[2].axhline(5.0, color='r', ls='--')
axes[2].axvspan(PLX - PLX_ERR, PLX + PLX_ERR, alpha=0.15, color='blue',
                label=r'$\varpi \pm 1\sigma$')
axes[2].set_xlabel(r'Parallax $\varpi$ (mas)')
axes[2].set_ylabel('$M_2$ (M$_\\odot$)')
axes[2].set_title('Parallax stress test')
axes[2].legend(fontsize=8)

fig.suptitle(f'Gaia DR3 {SOURCE_ID}', fontsize=11)
fig.tight_layout()
fpath = BASEDIR / 'paper' / 'figures' / 'fig_mass_posterior.pdf'
fpath.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fpath, dpi=200)
plt.close(fig)
print(f'  Figure → {fpath.name}')

# ── Save JSON ────────────────────────────────────────────────────────────
res = {
    'source_id': SOURCE_ID,
    'N_MC': N_MC,
    'nominal': {
        'M2_median': round(float(med), 1),
        'CI_68': [round(float(lo16), 1), round(float(hi84), 1)],
        'CI_90': [round(float(lo5), 1), round(float(hi95), 1)],
        'P_gt_5': round(float(P_gt5), 1),
        'P_gt_10': round(float(P_gt10), 1),
    },
    'inflated': {
        'plx_inflation': PLX_INFLATION,
        'M2_median': round(float(med_i), 1),
        'CI_68': [round(float(lo16_i), 1), round(float(hi84_i), 1)],
        'CI_90': [round(float(lo5_i), 1), round(float(hi95_i), 1)],
        'P_gt_5': round(float(P_gt5_i), 1),
        'P_gt_10': round(float(P_gt10_i), 1),
    },
    'parallax_stress': {
        'plx_plus_1sig_M2': round(float(m2_lo), 1),
        'plx_minus_1sig_M2': round(float(m2_hi), 1),
    },
    'note_posterior': (
        'Approximate stress test, not formal Bayesian posterior. '
        'Full NSS covariance matrix not publicly available. '
        'Parallax dominates the error budget (M_total ∝ plx^-3).'
    ),
}
jpath = BASEDIR / 'results' / 'mass_posterior_results.json'
jpath.parent.mkdir(parents=True, exist_ok=True)
jpath.write_text(json.dumps(res, indent=2))
print(f'  JSON  → {jpath.name}')
