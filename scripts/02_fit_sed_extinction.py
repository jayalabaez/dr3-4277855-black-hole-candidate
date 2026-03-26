#!/usr/bin/env python3
"""
02 — SED fitting, extinction, and primary-star self-consistency check
     for Gaia DR3 4277855016732107520.

Outputs:
  results/sed_fit_results.json
"""

import json, pathlib, numpy as np

# ── Target photometry ────────────────────────────────────────────────────
G_MAG   = 11.2495
BP_MAG  = 11.6566
RP_MAG  = 10.6638
BP_RP   = BP_MAG - RP_MAG          # 0.9928

TEFF    = 5922.0        # K  (GSP-Phot)
LOGG    = 2.932         # dex
M1      = 1.340         # M☉
PLX     = 1.5228        # mas
PLX_ERR = 0.1549
DIST    = 1e3 / PLX     # 656.7 pc

RSUN = 6.957e8          # m
LSUN = 3.828e26         # W
SB   = 5.670e-8         # Stefan-Boltzmann

# ── Extinction ───────────────────────────────────────────────────────────
BPRP_INTRINSIC = 0.76               # Pecaut & Mamajek 2013 for Teff≈5920K
E_BPRP = BP_RP - BPRP_INTRINSIC     # 0.233 mag
E_BV   = max(0.0, E_BPRP / 1.33)    # E(BP-RP)/E(B-V) ≈ 1.33
A_V    = E_BV * 3.1

EXT = {                              # A_λ / E(B-V), Fitzpatrick 1999
    'G': 2.74, 'BP': 3.37, 'RP': 2.04,
    'J': 0.87, 'H': 0.56,  'K': 0.35,
    'W1': 0.18, 'W2': 0.12,
}

# ── Photometric R and L ─────────────────────────────────────────────────
DM         = 5.0 * np.log10(DIST) - 5.0
A_G        = EXT['G'] * E_BV
M_G_dered  = G_MAG - A_G - DM

BC_G       = -0.30                   # approximate for Teff≈5920 K, evolved
M_bol      = M_G_dered + BC_G
L_phot     = 10**(0.4 * (4.74 - M_bol))
R_phot     = np.sqrt(L_phot * LSUN / (4 * np.pi * SB * TEFF**4)) / RSUN

# ── logg-based R and L ──────────────────────────────────────────────────
# R = sqrt(G M / 10^logg)  →  R/R☉ = sqrt(M1 / 10^(logg - 4.437))
R_logg = np.sqrt(M1 / 10**(LOGG - 4.437))
L_logg = R_logg**2 * (TEFF / 5778.0)**4

# ── Report ───────────────────────────────────────────────────────────────
print('=' * 70)
print('  SED / EXTINCTION / SELF-CONSISTENCY CHECK')
print('=' * 70)
print(f'  E(B-V)  = {E_BV:.3f}   A_V = {A_V:.3f}   DM = {DM:.2f}')
print(f'  M_G,dered = {M_G_dered:.2f}   M_bol = {M_bol:.2f}')
print()
print('  PHOTOMETRIC (distance + extinction):')
print(f'    R = {R_phot:.1f} R☉    L = {L_phot:.1f} L☉')
print()
print('  logg-BASED (logg + M1):')
print(f'    R = {R_logg:.1f} R☉    L = {L_logg:.1f} L☉')
print()

ratio_R = R_logg / R_phot
ratio_L = L_logg / L_phot
print(f'  DISCREPANCY:  R ratio = {ratio_R:.2f}x   L ratio = {ratio_L:.2f}x')
if abs(ratio_R - 1.0) > 0.25:
    print('  ⚠  Significant R/L inconsistency detected.')
    print('     GSP-Phot logg is unreliable for RUWE = 9.31 binaries.')
    print('     Photometric values adopted as fiducial; both reported.')
print()

# ── Save ─────────────────────────────────────────────────────────────────
res = {
    'source_id': 4277855016732107520,
    'E_BV': round(E_BV, 4),
    'A_V': round(A_V, 3),
    'DM': round(DM, 2),
    'M_G_dered': round(M_G_dered, 2),
    'M_bol': round(M_bol, 2),
    'L_photometric': round(L_phot, 1),
    'R_photometric': round(R_phot, 1),
    'L_logg': round(L_logg, 1),
    'R_logg': round(R_logg, 1),
    'R_discrepancy_factor': round(ratio_R, 2),
    'L_discrepancy_factor': round(ratio_L, 2),
    'note': 'GSP-Phot logg unreliable for RUWE=9.31 binary; photometric values fiducial.'
}
out = pathlib.Path(__file__).resolve().parent.parent / 'results' / 'sed_fit_results.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=2))
print(f'  Saved → {out.name}')
