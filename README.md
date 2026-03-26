# Gaia DR3 4277855016732107520 — Dark-Companion Candidate

> **Joel Ayala-Baez** | Independent Researcher  
> Paper status: Under review (MNRAS)

## Summary

We identify Gaia DR3 4277855016732107520 as a **high-priority spectroscopic follow-up target** from the Gaia DR3 non-single-star catalogue. The source carries a purely astrometric `Orbital` solution with no independent radial-velocity orbit; all inferred companion properties are **conditional** on the Gaia pipeline model.

**Important caveats:** ≳30–50% of comparable `Orbital` solutions fail independent RV validation. The stellar parameters rest on GSP-Phot values at RUWE = 9.31, a regime where systematic biases are known. Until RV confirmation is obtained, the companion mass and nature remain entirely provisional.

### Key Results (conditional on Gaia orbital solution)

| Property | Value | Note |
|---|---|---|
| Primary type | GSP-Phot evolved (unconfirmed) | No independent spectral classification |
| Period | 424.4 ± 1.2 d | Near-annual; scan-angle aliasing possible |
| Eccentricity | 0.343 ± 0.019 | |
| M₁ (adopted) | 1.34 ± 0.40 M☉ | PARSEC grid: median 1.09 [0.77, 1.62] |
| M₂ (median, ×1.7 inflated) | 12.3 M☉ | 90% CI [5.0, 35.9] M☉ |
| P(M₂ > 5 M☉ \| genuine) | 95% | Conditional on solution being genuine |
| MS companion exclusion | Excluded | 100–247× flux excess in optical |
| Stripped He star | Not favoured | NUV deficit 2700× |
| NSS significance | 75.4σ | |
| Predicted K₁ | ~65 km/s | Efficient RV confirmation |
| Distance | ~657 pc | |

## Repository Structure

```
dr3-4277855-black-hole-candidate/
├── paper/                            # MNRAS manuscript
│   ├── manuscript.tex
│   ├── manuscript.pdf
│   ├── references.bib
│   └── figures/                      # Publication figures (PDF)
├── scripts/                          # Modular analysis scripts
│   ├── 02_orbital_mc_posterior.py    # Monte Carlo mass posterior
│   ├── 03_primary_mass.py           # Primary mass estimation
│   ├── 04_companion_exclusion.py    # Band-by-band flux ratio test
│   ├── 05_hr_diagram.py             # HR diagram placement
│   ├── 06_orbital_geometry.py       # Orbital elements & geometry
│   ├── 07_uv_xray_radio.py         # Multi-wavelength limits
│   ├── 08_stripped_helium.py        # Stripped He star scan
│   ├── 09_sed_synthetic_phot.py     # Synthetic-photometry SED
│   ├── 10_orbital_candidate_comparison.py  # Population ranking
│   ├── 11_correlation_sensitivity.py      # Correlation tests
│   ├── 12_sed_synth_phot.py              # ATLAS9/PHOENIX SED
│   ├── 13_parsec_isochrone_m1.py         # PARSEC M1 grid
│   ├── 14_teff_chain_dependency.py       # Teff chain sensitivity
│   └── 15_recover_source_ids.py          # Source ID recovery
├── results/                          # Script outputs (JSON)
├── data/                             # Input data and metadata
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Reproducing the Analysis

All numerical results, figures, and tables in the manuscript are generated
from public Gaia DR3 catalogue inputs using the modular scripts above.
Each script reads its inputs from the Gaia Archive or the `data/` directory
and writes JSON results to `results/`.

```bash
# Clone
git clone https://github.com/jayalabaez/dr3-4277855-black-hole-candidate.git
cd dr3-4277855-black-hole-candidate

# Create environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run individual scripts (example)
python scripts/02_orbital_mc_posterior.py
python scripts/10_orbital_candidate_comparison.py
python scripts/13_parsec_isochrone_m1.py
```

Scripts 02–15 can be run independently in any order; each produces
self-contained JSON output under `results/`.

## Data Sources

- **Gaia DR3**: Astrometry, photometry, NSS orbital solution ([Gaia Archive](https://gea.esac.esa.int/archive/))
- **2MASS**: J, H, Ks photometry (via VizieR)
- **AllWISE**: W1, W2 photometry (via VizieR)
- **GALEX**: NUV photometry (via VizieR)
- **ROSAT / XMM-Newton**: X-ray upper limits (via VizieR)
- **SIMBAD**: Cross-identification

## Comparison with Confirmed Gaia BHs (contextual reference only)

These confirmed systems have independent multi-epoch RV orbits; this source has only a purely astrometric solution. The two classes of evidence are **not comparable**.

| System | M₂ (M☉) | Period (d) | Primary | Status |
|--------|----------|------------|---------|--------|
| Gaia BH1 | 9.6 | 185.6 | G-dwarf | Confirmed |
| Gaia BH2 | 8.9 | 1277 | RGB | Confirmed |
| Gaia BH3 | 33 | ~4200 | Giant | Confirmed |
| **This work** | **~12.3** | **424.4** | **GSP-Phot evolved** | **Candidate (unconfirmed)** |

## Citation

If you use this work, please cite:

```
Ayala-Baez, Joel (2026). Gaia DR3 4277855016732107520: A 12.3 M☉ Black Hole
Candidate from Astrometric Orbital Solution with RGB Primary.
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
