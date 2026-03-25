# Gaia DR3 4277855016732107520 — 12.3 M☉ Black Hole Candidate

> **Joel Ayala-Baez** | Independent Researcher  
> Paper status: Pre-submission draft

## Summary

Gaia DR3 4277855016732107520 is an astrometric binary harbouring a **12.3 M☉ dormant stellar-mass black hole candidate** — one of the strongest unconfirmed candidates in the Gaia era. The companion mass is a **true mass** from a full 3D astrometric orbital solution (not a minimum mass), eliminating the inclination degeneracy that limits spectroscopic-only candidates.

### Key Results

| Property | Value |
|---|---|
| Primary type | RGB (T_eff = 5922 K, log g = 2.93) |
| Period | 424.4 ± 1.2 d |
| Eccentricity | 0.343 ± 0.019 |
| M₁ (primary) | 1.34 ± 0.40 M☉ |
| M₂ (companion, TRUE mass) | 12.31 ± 7.52 M☉ |
| M₂ median (MC posterior) | 12.32 M☉ |
| M₂ 90% CI | [10.0, 15.4] M☉ |
| P(BH) | 100.0% |
| Companion light | Excluded (~30,000× threshold) |
| X-ray | Non-detection (dormant) |
| NSS significance | 75.4σ |
| RUWE | 9.31 |
| Alternative scenarios | 6/7 excluded, 1 disfavoured |
| Confirmation checklist | 13/13 PASS (GOLD tier) |
| Predicted K₁ | ~65 km/s |
| Distance | 656.7 pc |

## Repository Structure

```
dr3-4277855-black-hole-candidate/
├── data/                             # Input data and metadata
│   ├── dr3_4277855_input_summary.csv
│   ├── photometry_compiled.csv
│   └── metadata_notes.md
├── scripts/                          # Reproducibility scripts
│   └── dr3_4277855_publication_analysis.py  # Full 10-module pipeline
├── paper/                            # MNRAS manuscript
│   ├── manuscript.tex
│   ├── references.bib
│   ├── figures/                      # Publication figures (PNG)
│   │   ├── fig1_8panel_overview.png
│   │   ├── fig2_mass_posterior.png
│   │   ├── fig3_orbital_analysis.png
│   │   └── fig4_confirmation_checklist.png
│   └── tables/
├── results/                          # Script outputs (JSON)
│   └── dr3_4277855_summary.json
├── docs/                             # Additional documentation
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Quick Start

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

# Run full analysis pipeline (10 modules, ~30s)
python scripts/dr3_4277855_publication_analysis.py
```

## Analysis Modules

The single-script pipeline (`dr3_4277855_publication_analysis.py`) executes 10 modules:

| Module | Description |
|--------|-------------|
| A1 | Multi-archive data acquisition (Gaia + VizieR + SIMBAD) |
| A2 | Bayesian mass posterior with 500K MC draws |
| A3 | SED construction & extinction analysis (8-band) |
| A4 | Luminous companion exclusion (photometric) |
| A5 | Alternative scenario elimination (7 hypotheses) |
| A6 | Orbital dynamics & tidal circularization |
| A7 | SO(10) GUT theory context |
| A8 | Galactic context & kinematic population |
| A9 | Publication figure suite (4 figures) |
| A10 | Comprehensive report & JSON export |

## Data Sources

- **Gaia DR3**: Astrometry, photometry, NSS orbital solution ([Gaia Archive](https://gea.esac.esa.int/archive/))
- **2MASS**: J, H, Ks photometry (via VizieR)
- **AllWISE**: W1, W2 photometry (via VizieR)
- **GALEX**: NUV photometry (via VizieR)
- **ROSAT / XMM-Newton**: X-ray upper limits (via VizieR)
- **SIMBAD**: Cross-identification

## Comparison with Confirmed Gaia BHs

| System | M₂ (M☉) | Period (d) | Primary | Status |
|--------|----------|------------|---------|--------|
| Gaia BH1 | 9.6 | 185.6 | G-dwarf | Confirmed |
| Gaia BH2 | 8.9 | 1277 | RGB | Confirmed |
| Gaia BH3 | 33 | ~4200 | Giant | Confirmed |
| **This work** | **12.3** | **424.4** | **RGB** | **Candidate** |

## Citation

If you use this work, please cite:

```
Ayala-Baez, Joel (2026). Gaia DR3 4277855016732107520: A 12.3 M☉ Black Hole
Candidate from Astrometric Orbital Solution with RGB Primary.
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
