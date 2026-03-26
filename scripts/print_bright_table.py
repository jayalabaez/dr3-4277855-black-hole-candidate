import json

with open("scripts/outputs/orbital_bh_comparison.json") as f:
    data = json.load(f)

bright = [c for c in data["all_candidates"] if c["G"] < 12]
bright.sort(key=lambda x: x["K1_pred_kms"], reverse=True)

for i, c in enumerate(bright, 1):
    sid = c["source_id"]
    marker = " <--THIS" if "4277855" in sid else ""
    print(f"{i:2d}  {sid:>22s}  G={c['G']:5.2f}  dec={c['dec']:+6.1f}  "
          f"P={c['period_d']:7.1f}d  e={c['ecc']:.3f}  "
          f"M1={c['M1_msun']:5.2f}  M2={c['M2_msun']:5.1f}  "
          f"K1={c['K1_pred_kms']:5.1f}  sig={c['nss_sig']:5.1f}  "
          f"tier={c['tier']:6s}  acc={c['accessible']:4s}  "
          f"ann={c['annual_flag']}{marker}")
