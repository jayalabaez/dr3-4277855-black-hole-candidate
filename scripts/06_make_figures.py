#!/usr/bin/env python3
"""
06 — Publication figure suite for Gaia DR3 4277855016732107520.

Generates:
  paper/figures/fig_system_overview.pdf   (orbital schematic)
  paper/figures/fig_cmd_hrd.pdf           (CMD + HRD placement)
"""

import pathlib, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

BASEDIR = pathlib.Path(__file__).resolve().parent.parent
FIGDIR  = BASEDIR / 'paper' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ───────────────────────────────────────────────────────────
SOURCE_ID = 4277855016732107520
PERIOD    = 424.403      # d
ECC       = 0.3427
M1        = 1.340        # M☉
M2        = 12.313       # M☉
TEFF      = 5922.0       # K
a_AU      = 2.64         # AU
R_STAR    = 4.5          # R☉
L_STAR    = 22.0         # L☉
G_MAG     = 11.2495
BP_RP     = 0.9928
M_G       = 1.68         # dereddened absolute
BP_RP_0   = 0.76         # intrinsic

# ── Fig 1: system overview ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
theta = np.linspace(0, 2*np.pi, 500)
r = a_AU * (1 - ECC**2) / (1 + ECC * np.cos(theta))
x = r * np.cos(theta)
y = r * np.sin(theta)

ax.plot(x, y, 'k-', lw=1.5, alpha=0.7)
ax.plot(0, 0, 'ko', ms=14, label=f'Dark companion ({M2:.1f} M$_\\odot$)')
peri = a_AU * (1 - ECC)
apo  = a_AU * (1 + ECC)
ax.plot(peri, 0, 'o', color='orange', ms=10,
        label=f'Primary at periastron ({peri:.1f} AU)')

# Primary star (orange)
circle = Circle((peri, 0), 0.08, color='orange', alpha=0.7)
ax.add_patch(circle)

ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.set_title(f'Gaia DR3 {SOURCE_ID}\n'
             f'P = {PERIOD:.1f} d, e = {ECC:.3f}, a = {a_AU:.2f} AU',
             fontsize=11)
ax.set_aspect('equal')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Annotation
ax.annotate(f'Periastron: {peri:.1f} AU\nApastron: {apo:.1f} AU',
            xy=(0.02, 0.98), xycoords='axes fraction', va='top',
            fontsize=8, bbox=dict(boxstyle='round', fc='lightyellow'))

fig.tight_layout()
fig.savefig(FIGDIR / 'fig_system_overview.pdf', dpi=200)
plt.close(fig)
print(f'  fig_system_overview.pdf')

# ── Fig 2: CMD + HRD ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# CMD
ax1.plot(BP_RP_0, M_G, 'r*', ms=15, zorder=5, label='This source')

# Rough MS locus
ms_bprp = np.array([-0.3, 0.0, 0.3, 0.6, 0.82, 1.0, 1.3, 1.5, 2.0, 2.5])
ms_MG   = np.array([-1.0, 0.5, 2.0, 3.5, 4.5,  5.5, 7.0, 8.0, 10.0, 12.0])
ax1.plot(ms_bprp, ms_MG, 'b-', alpha=0.4, lw=6, label='MS locus (approx.)')

# RGB locus
rgb_bprp = np.array([0.7, 0.8, 1.0, 1.2, 1.5])
rgb_MG   = np.array([2.5, 1.5, 0.5, -0.5, -1.5])
ax1.plot(rgb_bprp, rgb_MG, '-', color='orangered', alpha=0.4, lw=6,
         label='RGB locus (approx.)')

ax1.invert_yaxis()
ax1.set_xlabel(r'$(G_{\rm BP} - G_{\rm RP})_0$')
ax1.set_ylabel(r'$M_G$ (mag)')
ax1.set_title('Colour–Magnitude Diagram')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# HRD
ax2.plot(np.log10(TEFF), np.log10(L_STAR), 'r*', ms=15, zorder=5,
         label='This source')

# MS approx
ms_teff = np.array([30000, 15000, 9000, 7000, 6000, 5500, 5000, 4000, 3500])
ms_L    = np.array([30000, 1000,  30,   8,    2,    1,    0.5,  0.1,  0.02])
ax2.plot(np.log10(ms_teff), np.log10(ms_L), 'b-', alpha=0.4, lw=6,
         label='MS (approx.)')

# RGB approx
rgb_teff = np.array([5500, 5200, 5000, 4500, 4000])
rgb_L    = np.array([5,    15,   50,   200,  500])
ax2.plot(np.log10(rgb_teff), np.log10(rgb_L), '-', color='orangered',
         alpha=0.4, lw=6, label='RGB (approx.)')

ax2.invert_xaxis()
ax2.set_xlabel(r'$\log\,T_{\rm eff}$ (K)')
ax2.set_ylabel(r'$\log\,L/L_\odot$')
ax2.set_title('Hertzsprung–Russell Diagram')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

fig.suptitle(f'Gaia DR3 {SOURCE_ID}', fontsize=11)
fig.tight_layout()
fig.savefig(FIGDIR / 'fig_cmd_hrd.pdf', dpi=200)
plt.close(fig)
print(f'  fig_cmd_hrd.pdf')
