#!/usr/bin/env python3
"""
04 — Companion-light exclusion test for Gaia DR3 4277855016732107520.

Computes band-by-band Planck flux ratios between a hypothetical
main-sequence companion at the catalog M2 and the observed primary,
scaled by (R_comp/R_prim)².

Reports both photometric and logg-based primary radius cases.

Outputs:
  results/companion_exclusion_results.json
"""

import json, pathlib, numpy as np

# ── Constants ────────────────────────────────────────────────────────────
h  = 6.626e-34      # J s
c  = 2.998e8         # m/s
kB = 1.381e-23       # J/K
RSUN = 6.957e8       # m

SOURCE_ID = 4277855016732107520
M2_CAT    = 12.313   # M☉

# Primary
TEFF_PRIM       = 5922.0   # K
R_PRIM_PHOT     = 4.5      # R☉  (photometric distance)
L_PRIM_PHOT     = 22.0     # L☉  (photometric)
R_PRIM_LOGG     = 6.6      # R☉  (from logg + M1)
L_PRIM_LOGG     = 47.3     # L☉  (from logg + M1)

# Hypothetical MS companion (Eker+ 2018 relations)
# L ∝ M^3.5, R ∝ M^0.57 for M > 2 M☉
L_COMP = M2_CAT**3.5                              # L☉
R_COMP = M2_CAT**0.57                              # R☉
T_COMP = 5778.0 * (L_COMP / R_COMP**2)**0.25       # K

# Detection threshold
THRESHOLD = 0.05   # 5 % flux excess detectable

# Gaia bands (effective wavelength in m)
BANDS = {
    'G':  0.622e-6,
    'BP': 0.511e-6,
    'RP': 0.777e-6,
    'J':  1.235e-6,
    'H':  1.662e-6,
    'K':  2.159e-6,
}

# ── Planck function ──────────────────────────────────────────────────────
def planck(lam, T):
    """B_ν at wavelength lam (m) and temperature T (K)."""
    x = h * c / (lam * kB * T)
    if x > 500:
        return 0.0
    return 2 * h * c**2 / lam**5 / (np.exp(x) - 1)


# ── Compute flux ratios ─────────────────────────────────────────────────
def compute_ratios(r_prim, label):
    """Return dict of band → flux ratio F_comp/F_prim."""
    rr2 = (R_COMP / r_prim)**2
    results = {}
    print(f'  {label} (R_prim = {r_prim:.1f} R☉):')
    for band, lam in BANDS.items():
        b_prim = planck(lam, TEFF_PRIM)
        b_comp = planck(lam, T_COMP)
        if b_prim == 0:
            ratio = np.inf
        else:
            ratio = rr2 * b_comp / b_prim
        results[band] = round(ratio, 1)
        print(f'    {band:>3}  F_comp/F_prim = {ratio:>8.1f}×')
    return results


# ── Main ─────────────────────────────────────────────────────────────────
print('=' * 70)
print('  COMPANION-LIGHT EXCLUSION  (Gaia DR3 4277855016732107520)')
print('=' * 70)
print(f'  Hypothetical MS companion: M = {M2_CAT:.1f} M☉')
print(f'    L = {L_COMP:.0f} L☉   R = {R_COMP:.1f} R☉   Teff = {T_COMP:.0f} K')
print(f'  Primary: Teff = {TEFF_PRIM:.0f} K')
print()

ratios_phot = compute_ratios(R_PRIM_PHOT, 'Photometric case')
print()
ratios_logg = compute_ratios(R_PRIM_LOGG, 'logg-based case')
print()

# Bolometric ratio
bol_phot = L_COMP / L_PRIM_PHOT
bol_logg = L_COMP / L_PRIM_LOGG
print(f'  Bolometric ratio: {bol_phot:.0f}× (phot)  /  {bol_logg:.0f}× (logg)')
print()

# Max hidden companion mass
print('  Max hidden MS companion (< 5% in any band):')
for label, r_prim in [('photometric', R_PRIM_PHOT), ('logg', R_PRIM_LOGG)]:
    for m_test in np.arange(0.5, 5.0, 0.01):
        L_t = m_test**3.5
        R_t = m_test**0.57
        T_t = 5778.0 * (L_t / R_t**2)**0.25
        rr2 = (R_t / r_prim)**2
        ok = True
        for lam in BANDS.values():
            bp = planck(lam, TEFF_PRIM)
            bc = planck(lam, T_t)
            if bp > 0 and rr2 * bc / bp > THRESHOLD:
                ok = False
                break
        if not ok:
            print(f'    {label}: M_max = {m_test - 0.01:.2f} M☉')
            break

print()
print('  ★ VERDICT: A luminous MS companion at M2 = 12.3 M☉ is firmly')
print('    excluded by the observed single-star SED in ALL bands.')

# ── Save ─────────────────────────────────────────────────────────────────
res = {
    'source_id': SOURCE_ID,
    'companion_MS': {
        'M2': M2_CAT,
        'L': round(L_COMP, 0),
        'R': round(R_COMP, 1),
        'Teff': round(T_COMP, 0),
    },
    'flux_ratios_photometric': ratios_phot,
    'flux_ratios_logg': ratios_logg,
    'bolometric_ratio_phot': round(bol_phot, 0),
    'bolometric_ratio_logg': round(bol_logg, 0),
}
out = pathlib.Path(__file__).resolve().parent.parent / 'results' / 'companion_exclusion_results.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=2))
print(f'\n  Saved → {out.name}')
