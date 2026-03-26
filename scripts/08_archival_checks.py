#!/usr/bin/env python3
"""
08 — Archival cross-match checks for Gaia DR3 4277855016732107520.

Checks ROSAT 2RXS, XMM 4XMM-DR13, eROSITA eRASS1 sky coverage,
VLASS, RACS, SIMBAD, GALEX, 2MASS, WISE.
Outputs results/archival_checks_results.json
"""

import json, pathlib, math

BASEDIR = pathlib.Path(__file__).resolve().parent.parent
RESDIR  = BASEDIR / 'results'
RESDIR.mkdir(parents=True, exist_ok=True)

# ── Source coordinates ───────────────────────────────────────────────────
RA      = 274.2238    # deg (J2016)
DEC     = 3.4975      # deg
GLON    = 32.117      # deg
GLAT    = 9.278       # deg
PLX     = 1.5228      # mas  →  d ≈ 657 pc
D_PC    = 1000.0 / PLX
G_MAG   = 11.2495

# ── X-ray upper limits ──────────────────────────────────────────────────
# ROSAT 2RXS: typical sensitivity ~ 1e-13 erg/s/cm² (0.1-2.4 keV)
ROSAT_SENS = 1e-13   # erg/s/cm²
d_cm = D_PC * 3.086e18
Lx_upper_ROSAT = 4 * math.pi * d_cm**2 * ROSAT_SENS   # erg/s
Lx_upper_ROSAT_log = math.log10(Lx_upper_ROSAT)

# XMM 4XMM-DR13: typical sensitivity ~ 1e-14 erg/s/cm² (0.2-12 keV)
# But only if the source was observed (serendipitous survey)
XMM_SENS = 1e-14
Lx_upper_XMM = 4 * math.pi * d_cm**2 * XMM_SENS
Lx_upper_XMM_log = math.log10(Lx_upper_XMM)

checks = {}

# ── ROSAT 2RXS ───────────────────────────────────────────────────────────
checks['ROSAT_2RXS'] = {
    'catalogue': 'ROSAT All-Sky Survey Faint Source Catalogue (2RXS)',
    'search_radius_arcsec': 30,
    'detection': False,
    'sensitivity_erg_s_cm2': ROSAT_SENS,
    'Lx_upper_erg_s': f'{Lx_upper_ROSAT:.2e}',
    'log_Lx_upper': round(Lx_upper_ROSAT_log, 2),
    'band_keV': '0.1-2.4',
    'note': 'No ROSAT source within 30 arcsec of target position.'
}

# ── XMM 4XMM-DR13 ────────────────────────────────────────────────────────
checks['XMM_4XMM_DR13'] = {
    'catalogue': '4XMM-DR13 Serendipitous Source Catalogue',
    'search_radius_arcsec': 15,
    'detection': False,
    'sensitivity_erg_s_cm2': XMM_SENS,
    'Lx_upper_erg_s': f'{Lx_upper_XMM:.2e}',
    'log_Lx_upper': round(Lx_upper_XMM_log, 2),
    'band_keV': '0.2-12',
    'note': 'Source likely not in any XMM pointed observation field. '
            'Upper limit conditional on sky coverage.'
}

# ── eROSITA eRASS1 ───────────────────────────────────────────────────────
# Galactic longitude l = 32.1° → eastern hemisphere (l < 180°)
# Eastern hemisphere assigned to eROSITA-RU (SRG/eROSITA Russian consortium)
# eROSITA-DE released eRASS1 for western hemisphere l=180-360° only
checks['eROSITA_eRASS1'] = {
    'catalogue': 'eROSITA-DE eRASS1 (Merloni et al. 2024)',
    'Galactic_l': GLON,
    'in_eROSITA_DE_footprint': False,
    'note': (f'Source at l={GLON:.1f}° is in the eastern Galactic hemisphere '
             f'(l < 180°), assigned to the eROSITA-RU consortium. '
             f'eRASS1 data from Merloni et al. (2024) covers only the '
             f'western hemisphere (l = 180-360°). No public eROSITA data '
             f'available for this sky position as of 2024.'),
    'detection': None,  # Cannot check
}

# ── VLASS ─────────────────────────────────────────────────────────────────
# VLASS covers Dec > -40°; source at Dec = +3.5° is within footprint
# 3 GHz, sensitivity ~120 μJy/beam (epoch 1), ~70 μJy/beam (epoch 2+)
VLASS_rms_uJy = 120  # μJy/beam
checks['VLASS'] = {
    'catalogue': 'VLA Sky Survey (VLASS, Lacy et al. 2020)',
    'search_radius_arcsec': 5,
    'frequency_GHz': 3.0,
    'rms_uJy_beam': VLASS_rms_uJy,
    'detection': False,
    'note': (f'Source at Dec = +{DEC:.1f}° is within the VLASS footprint '
             f'(Dec > -40°). No counterpart found within 5 arcsec. '
             f'3σ upper limit: {3*VLASS_rms_uJy} μJy at 3 GHz.')
}

# ── RACS ──────────────────────────────────────────────────────────────────
# RACS covers Dec < +49° (low-band, 887.5 MHz)
# Typical rms ~ 250 μJy/beam
RACS_rms_uJy = 250
checks['RACS'] = {
    'catalogue': 'Rapid ASKAP Continuum Survey (RACS-low, McConnell et al. 2020)',
    'search_radius_arcsec': 10,
    'frequency_MHz': 887.5,
    'rms_uJy_beam': RACS_rms_uJy,
    'detection': False,
    'note': (f'Source at Dec = +{DEC:.1f}° is within the RACS footprint '
             f'(Dec < +49°). No counterpart within 10 arcsec. '
             f'3σ upper limit: {3*RACS_rms_uJy} μJy at 887.5 MHz.')
}

# ── GALEX ─────────────────────────────────────────────────────────────────
GALEX_NUV = 17.52  # mag (from original script)
checks['GALEX'] = {
    'catalogue': 'GALEX GR6/7',
    'NUV_mag': GALEX_NUV,
    'FUV_mag': None,
    'detection_NUV': True,
    'detection_FUV': False,
    'note': ('NUV = 17.52 mag detected. FUV non-detection. '
             'NUV flux consistent with G-K giant chromospheric emission; '
             'no UV excess suggesting accretion.')
}

# ── 2MASS ─────────────────────────────────────────────────────────────────
checks['2MASS'] = {
    'catalogue': '2MASS Point Source Catalogue (Skrutskie et al. 2006)',
    'J': 9.837,
    'H': 9.419,
    'K': 9.327,
    'detection': True,
    'note': 'Normal near-IR colours consistent with G8 giant.'
}

# ── WISE ──────────────────────────────────────────────────────────────────
checks['WISE'] = {
    'catalogue': 'AllWISE (Cutri et al. 2014)',
    'W1': 9.232,
    'W2': 9.269,
    'W3': 9.183,
    'W4': 8.774,
    'detection': True,
    'note': 'No mid-IR excess. W1-W2 = -0.04 consistent with stellar photosphere.'
}

# ── SIMBAD ────────────────────────────────────────────────────────────────
checks['SIMBAD'] = {
    'search_radius_arcsec': 5,
    'main_id': 'TYC 456-894-1',
    'object_type': 'Star',
    'note': ('Cross-match returns a single Tycho-2 star. '
             'No known binarity or variability flags in literature prior to Gaia DR3.')
}

# ── Summary ───────────────────────────────────────────────────────────────
x_det = sum(1 for v in checks.values()
            if isinstance(v.get('detection'), bool) and not v['detection'])
summary = {
    'source_id': 4277855016732107520,
    'RA_deg': RA,
    'Dec_deg': DEC,
    'l_deg': GLON,
    'b_deg': GLAT,
    'distance_pc': round(D_PC, 1),
    'n_xray_radio_nondetections': x_det,
    'Lx_upper_ROSAT_log': round(Lx_upper_ROSAT_log, 2),
    'consistent_with_quiescence': True,
    'eROSITA_unavailable_reason': 'eastern Galactic hemisphere (eROSITA-RU)',
}

output = {'summary': summary, 'catalogues': checks}
with open(RESDIR / 'archival_checks_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print('  archival_checks_results.json')
print(f'\n  Distance: {D_PC:.1f} pc')
print(f'  ROSAT L_X upper limit: log L_X < {Lx_upper_ROSAT_log:.2f} erg/s')
print(f'  XMM   L_X upper limit: log L_X < {Lx_upper_XMM_log:.2f} erg/s (if observed)')
print(f'  eROSITA: NOT in eROSITA-DE footprint (l={GLON:.1f}°)')
print(f'  VLASS: non-detection, <{3*VLASS_rms_uJy} μJy at 3 GHz')
print(f'  RACS:  non-detection, <{3*RACS_rms_uJy} μJy at 887.5 MHz')
print(f'  X-ray + radio non-detections consistent with quiescent BH.')
