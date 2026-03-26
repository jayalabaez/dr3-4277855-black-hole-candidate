#!/usr/bin/env python3
"""
07 — Sensitivity analysis for Gaia DR3 4277855016732107520.

Varies M1 and parallax inflation, reports M2 posterior summaries.
Outputs results/sensitivity_results.json  +  paper/figures/fig_sensitivity.pdf
"""

import json, pathlib, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASEDIR = pathlib.Path(__file__).resolve().parent.parent
FIGDIR  = BASEDIR / 'paper' / 'figures'
RESDIR  = BASEDIR / 'results'
for d in (FIGDIR, RESDIR):
    d.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
N = 200_000

# ── Catalog values ───────────────────────────────────────────────────────
M_TOTAL_CAT = 13.653    # M1+M2 from catalog (1.340 + 12.313)
PERIOD      = 424.403   # d
PERIOD_ERR  = 1.159     # d
PLX         = 1.5228    # mas
PLX_ERR     = 0.1549    # mas

M1_GRID = [1.0, 1.1, 1.2, 1.34, 1.5, 1.7, 2.0]
PLX_INF = [1.0, 1.3, 1.5, 1.7, 2.0]

def run_mc(m1, plx_inflation):
    plx_sigma = PLX_ERR * plx_inflation
    plx_draws = np.random.normal(PLX, plx_sigma, N)
    plx_draws = plx_draws[plx_draws > 0.3]
    P_draws   = np.random.normal(PERIOD, PERIOD_ERR, len(plx_draws))
    P_draws   = P_draws[P_draws > 50]
    n = min(len(plx_draws), len(P_draws))
    plx_draws = plx_draws[:n]
    P_draws   = P_draws[:n]

    sys_fac = np.random.uniform(0.95, 1.05, n)
    m_total = M_TOTAL_CAT * sys_fac * (PLX / plx_draws)**3 * (PERIOD / P_draws)**2
    m2 = m_total - m1
    m2 = m2[m2 > 0]
    return m2

# ── Main grid ────────────────────────────────────────────────────────────
rows = []
for inf in PLX_INF:
    for m1 in M1_GRID:
        m2 = run_mc(m1, inf)
        med = float(np.median(m2))
        lo  = float(np.percentile(m2, 16))
        hi  = float(np.percentile(m2, 84))
        p5  = float(np.mean(m2 > 5) * 100)
        p10 = float(np.mean(m2 > 10) * 100)
        rows.append({
            'M1': m1, 'plx_inflation': inf,
            'M2_median': round(med, 2),
            'M2_16': round(lo, 2), 'M2_84': round(hi, 2),
            'P_gt5': round(p5, 1), 'P_gt10': round(p10, 1)
        })

with open(RESDIR / 'sensitivity_results.json', 'w') as f:
    json.dump(rows, f, indent=2)
print('  sensitivity_results.json')

# ── Figure: heat-map of P(>5 M☉) ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for ax, key, title in [(ax1, 'P_gt5', r'$P(M_2 > 5\,M_\odot)$ (%)'),
                         (ax2, 'P_gt10', r'$P(M_2 > 10\,M_\odot)$ (%)')]:
    grid = np.zeros((len(PLX_INF), len(M1_GRID)))
    for r in rows:
        i = PLX_INF.index(r['plx_inflation'])
        j = M1_GRID.index(r['M1'])
        grid[i, j] = r[key]

    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[M1_GRID[0]-0.05, M1_GRID[-1]+0.05,
                           PLX_INF[0]-0.05, PLX_INF[-1]+0.05],
                   cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xlabel(r'$M_1$ ($M_\odot$)')
    ax.set_ylabel(r'Parallax inflation factor')
    ax.set_title(title)
    ax.set_xticks(M1_GRID)
    ax.set_yticks(PLX_INF)
    fig.colorbar(im, ax=ax, shrink=0.8)

    for r in rows:
        i = PLX_INF.index(r['plx_inflation'])
        j = M1_GRID.index(r['M1'])
        ax.text(r['M1'], r['plx_inflation'], f"{r[key]:.0f}",
                ha='center', va='center', fontsize=7, fontweight='bold')

fig.suptitle(f'Gaia DR3 4277855016732107520 — Sensitivity Analysis', fontsize=11)
fig.tight_layout()
fig.savefig(FIGDIR / 'fig_sensitivity.pdf', dpi=200)
plt.close(fig)
print('  fig_sensitivity.pdf')

# ── Console summary (nominal M1=1.34, inflation=1.7) ────────────────────
nom = [r for r in rows if r['M1']==1.34 and r['plx_inflation']==1.7][0]
print(f"\n  Nominal (M1=1.34, plx_inf=1.7):")
print(f"    M2 = {nom['M2_median']:.2f}  [{nom['M2_16']:.2f}, {nom['M2_84']:.2f}]")
print(f"    P(>5) = {nom['P_gt5']:.1f}%,  P(>10) = {nom['P_gt10']:.1f}%")
