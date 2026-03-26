#!/usr/bin/env python3
"""
10_orbital_candidate_comparison.py

Compute predicted K1 (edge-on upper bound) for all Orbital BH candidates
in the gravitas_omniscan_v13 catalog and rank them by follow-up efficiency
metrics (K1, brightness, declination accessibility).

Output:  scripts/outputs/orbital_bh_comparison.json
"""

import csv
import json
import math
import os

# Constants
G_SI = 6.67430e-11        # m^3 kg^-1 s^-2
M_SUN = 1.98892e30        # kg
DAY_S = 86400.0

CATALOG = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "outputs", "gravitas_omniscan_v13",
    "gravitas_omniscan_catalog_v13.csv",
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
OUTPUT = os.path.join(OUTPUT_DIR, "orbital_bh_comparison.json")

THIS_SOURCE_PREFIX = "4277855"  # first 7 digits of 4277855016732107520


def compute_K1_edge_on(M1_msun, M2_msun, P_days, ecc):
    """Predicted primary RV semi-amplitude assuming sin(i) = 1 (edge-on).

    K1 = (2*pi*G / P)^(1/3) * M2 / (M1+M2)^(2/3) / sqrt(1-e^2)
    """
    P_s = P_days * DAY_S
    M1 = M1_msun * M_SUN
    M2 = M2_msun * M_SUN
    M_tot = M1 + M2

    K1 = (
        (2 * math.pi * G_SI / P_s) ** (1.0 / 3.0)
        * M2
        / M_tot ** (2.0 / 3.0)
        / math.sqrt(1 - ecc ** 2)
    )
    return K1 / 1e3  # km/s


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    candidates = []
    with open(CATALOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sol_type"] != "Orbital" or row["cat"] != "BH":
                continue

            sid = row["source_id"].replace(".", "").split("e+")[0]
            # reconstruct full source_id from scientific notation
            val = float(row["source_id"])
            sid_int = str(int(val))

            M1 = float(row["M1"])
            M2 = float(row["M2"])
            P = float(row["period"])
            ecc = float(row["ecc"])
            G_mag = float(row["G"])
            dec = float(row["dec"])
            sig = float(row["sig"])
            gof = float(row["gof"])
            ruwe = float(row["ruwe"])
            tier = row["tier"]
            P_BH = float(row["P_BH"]) if row["P_BH"] else 0.0

            K1_pred = compute_K1_edge_on(M1, M2, P, ecc)

            # Accessibility: |dec| < 60 => both hemispheres
            accessible = "Both" if abs(dec) < 45 else ("N" if dec > 0 else "S")

            # Near-annual period flag (0.8-1.5 yr)
            P_yr = P / 365.25
            annual_flag = 0.8 < P_yr < 1.5

            candidates.append({
                "source_id": sid_int,
                "G": round(G_mag, 2),
                "dec": round(dec, 1),
                "period_d": round(P, 1),
                "ecc": round(ecc, 3),
                "M1_msun": round(M1, 2),
                "M2_msun": round(M2, 1),
                "K1_pred_kms": round(K1_pred, 1),
                "nss_sig": round(sig, 1),
                "GOF": round(gof, 1),
                "RUWE": round(ruwe, 2),
                "tier": tier,
                "P_BH": round(P_BH, 3),
                "accessible": accessible,
                "annual_flag": annual_flag,
            })

    # Sort by K1 descending
    candidates.sort(key=lambda x: x["K1_pred_kms"], reverse=True)

    # Find this source
    this = [c for c in candidates if c["source_id"].startswith(THIS_SOURCE_PREFIX)]
    rank_k1 = next(
        (i + 1 for i, c in enumerate(candidates)
         if c["source_id"].startswith(THIS_SOURCE_PREFIX)),
        None,
    )

    # Find bright (G < 12) candidates
    bright = [c for c in candidates if c["G"] < 12.0]
    bright_k1 = sorted(bright, key=lambda x: x["K1_pred_kms"], reverse=True)

    # Top 10 by K1
    top10 = candidates[:10]

    # Subset for paper table: bright + high K1 or GOLD tier
    paper_subset = [c for c in candidates
                    if c["G"] < 13.0 and c["K1_pred_kms"] > 20]

    results = {
        "total_orbital_bh": len(candidates),
        "this_source": this[0] if this else None,
        "rank_by_K1": rank_k1,
        "rank_among_bright_G12": next(
            (i + 1 for i, c in enumerate(bright_k1)
             if c["source_id"].startswith(THIS_SOURCE_PREFIX)), None),
        "n_bright_G12": len(bright),
        "top10_by_K1": top10,
        "paper_table_candidates": paper_subset,
        "all_candidates": candidates,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Total Orbital BH candidates: {len(candidates)}")
    print(f"This source rank by K1: {rank_k1} / {len(candidates)}")
    print(f"This source K1: {this[0]['K1_pred_kms'] if this else 'N/A'} km/s")
    print(f"Bright (G<12) candidates: {len(bright)}")
    print(f"This source rank among bright: "
          f"{results['rank_among_bright_G12']} / {len(bright)}")
    print()
    print("Top 10 by predicted K1 (edge-on):")
    print(f"{'Rank':>4} {'Source ID':>20} {'G':>6} {'Dec':>6} {'P(d)':>7} "
          f"{'M2':>5} {'K1':>6} {'Sig':>6} {'Tier':>7}")
    for i, c in enumerate(top10):
        marker = " <--" if c["source_id"].startswith(THIS_SOURCE_PREFIX) else ""
        print(f"{i+1:4d} {c['source_id']:>20} {c['G']:6.2f} {c['dec']:6.1f} "
              f"{c['period_d']:7.1f} {c['M2_msun']:5.1f} "
              f"{c['K1_pred_kms']:6.1f} {c['nss_sig']:6.1f} "
              f"{c['tier']:>7}{marker}")


if __name__ == "__main__":
    main()
