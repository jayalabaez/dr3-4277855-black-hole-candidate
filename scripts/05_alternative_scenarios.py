#!/usr/bin/env python3
"""
05 — Alternative scenario assessment for Gaia DR3 4277855016732107520.

Uses 4-tier verdict system: Ruled out / Strongly disfavoured /
Not favoured / Requires follow-up.

Key difference from AstroSpectroSB1 targets: this is an Orbital-only
solution — artefact/alignment scenarios CANNOT be excluded without
independent RV confirmation.

Outputs:
  results/alternative_scenarios_results.json
"""

import json, pathlib

SOURCE_ID  = 4277855016732107520
M2         = 12.313      # M☉
M1         = 1.340       # M☉
PERIOD     = 424.403     # d
ECC        = 0.3427
L_PRIM     = 22.0        # L☉ (photometric)
SOL_TYPE   = 'Orbital'

scenarios = []

# ── 1. Main-sequence companion ───────────────────────────────────────────
L_ms = M2**3.5
scenarios.append({
    'scenario': 'Main-sequence companion',
    'verdict': 'RULED OUT',
    'reason': (f'A {M2:.1f} M☉ MS star would have L ≈ {L_ms:.0f} L☉ '
               f'(Teff ≈ 24,000 K), dominating the SED in all optical/IR bands. '
               f'No secondary component detected. SED is consistent with '
               f'single evolved star.')
})

# ── 2. White dwarf ──────────────────────────────────────────────────────
M_Ch = 1.44
scenarios.append({
    'scenario': 'White dwarf',
    'verdict': 'RULED OUT',
    'reason': (f'M2 = {M2:.1f} M☉ exceeds the Chandrasekhar limit '
               f'({M_Ch} M☉) by {M2/M_Ch:.0f}×.')
})

# ── 3. Neutron star ─────────────────────────────────────────────────────
M_TOV = 2.3
scenarios.append({
    'scenario': 'Neutron star',
    'verdict': 'RULED OUT',
    'reason': (f'M2 = {M2:.1f} M☉ exceeds the TOV limit '
               f'({M_TOV} M☉) by {M2/M_TOV:.0f}×.')
})

# ── 4. Hierarchical triple ──────────────────────────────────────────────
M_each = M2 / 2
L_each = M_each**3.5
L_triple = 2 * L_each
P_inner_max = PERIOD / 4.7 * (1 - ECC)**1.8
scenarios.append({
    'scenario': 'Hierarchical triple',
    'verdict': 'STRONGLY DISFAVOURED',
    'reason': (f'Two {M_each:.1f} M☉ MS stars: combined L ≈ {L_triple:.0f} L☉ '
               f'vs. primary L ≈ {L_PRIM:.0f} L☉. Would dominate SED. '
               f'Stability requires P_inner < {P_inner_max:.0f} d. '
               f'No composite spectrum detected. However, a triple with '
               f'two compact objects (e.g. NS+NS) cannot be excluded by '
               f'photometry alone.')
})

# ── 5. Stripped helium star ──────────────────────────────────────────────
scenarios.append({
    'scenario': 'Stripped helium star / hot subdwarf',
    'verdict': 'NOT FAVOURED',
    'reason': (f'A {M2:.1f} M☉ stripped He star would have '
               f'Teff > 50,000 K and L > 10,000 L☉. '
               f'Expected strong UV excess: predicted NUV ≈ 9 vs '
               f'observed 17.5 (>2000× fainter). '
               f'Not favoured but requires UV spectroscopy for firm exclusion.')
})

# ── 6. Astrometric artefact ─────────────────────────────────────────────
scenarios.append({
    'scenario': 'Astrometric artefact (spurious Gaia solution)',
    'verdict': 'REQUIRES FOLLOW-UP',
    'reason': (f'Solution type is {SOL_TYPE} (astrometry only, no independent '
               f'spectroscopic confirmation). Recent RV follow-up studies show '
               f'many Gaia DR3 Orbital BH candidates fail spectroscopic checks. '
               f'NSS significance ({75.4}σ) and RUWE (9.31) are supportive but '
               f'not sufficient to exclude scan-angle biases or marginally '
               f'resolved companion effects. '
               f'Requires independent multi-epoch RV orbit.')
})

# ── 7. Chance alignment ─────────────────────────────────────────────────
scenarios.append({
    'scenario': 'Chance alignment / blend-driven spurious solution',
    'verdict': 'REQUIRES FOLLOW-UP',
    'reason': (f'At G = 11.25, source confusion is low but at b = 9.3° '
               f'blending is not negligible. A coherent {PERIOD:.0f}-d Keplerian '
               f'signal at 75σ argues against pure noise, but the absence of '
               f'independent RV confirmation means this cannot be fully excluded. '
               f'Requires spectroscopic follow-up.')
})

# ── Summary ──────────────────────────────────────────────────────────────
print('=' * 70)
print('  ALTERNATIVE SCENARIO ASSESSMENT')
print('=' * 70)
for i, s in enumerate(scenarios, 1):
    print(f'\n  [{i}] {s["scenario"]}')
    print(f'      Verdict: {s["verdict"]}')
    print(f'      {s["reason"][:120]}...' if len(s["reason"]) > 120 else f'      {s["reason"]}')

n_ruled   = sum(1 for s in scenarios if s['verdict'] == 'RULED OUT')
n_strong  = sum(1 for s in scenarios if s['verdict'] == 'STRONGLY DISFAVOURED')
n_not_fav = sum(1 for s in scenarios if s['verdict'] == 'NOT FAVOURED')
n_follow  = sum(1 for s in scenarios if s['verdict'] == 'REQUIRES FOLLOW-UP')

print(f'\n  Summary: {n_ruled} ruled out, {n_strong} strongly disfavoured, '
      f'{n_not_fav} not favoured, {n_follow} require follow-up')
print()

# ── Save ─────────────────────────────────────────────────────────────────
out = pathlib.Path(__file__).resolve().parent.parent / 'results' / 'alternative_scenarios_results.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    'source_id': SOURCE_ID,
    'solution_type': SOL_TYPE,
    'scenarios': scenarios,
    'summary': {
        'ruled_out': n_ruled,
        'strongly_disfavoured': n_strong,
        'not_favoured': n_not_fav,
        'requires_followup': n_follow,
    }
}, indent=2))
print(f'  Saved → {out.name}')
