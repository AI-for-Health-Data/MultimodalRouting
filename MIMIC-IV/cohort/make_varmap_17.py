#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_varmap_17.py

Builds a variable map CSV for 17 common ICU variables (7 vitals, 10 labs)
for MIMIC-IV, matching the schema expected by the cohort builder:

Columns:
  variable, source, itemid, priority, unit, to_unit

Notes:
- source ∈ {"chartevents","labevents"}
- priority: 1 = highest; MetaVision itemids (>=220000) preferred over CareVue
- unit is taken from the source tables; Temperature has to_unit='c' so downstream code converts F→C

Usage:
  python make_varmap_17.py \
      --d_items d_items.csv.gz \
      --d_labitems d_labitems.csv.gz \
      --out varmap_mimiciv_17.csv \
      --max_per_var 8
"""

import argparse
import re
import sys
import pandas as pd

def read_tables(d_items_path: str, d_labitems_path: str):
    d_items = pd.read_csv(d_items_path, compression="gzip", low_memory=False)
    d_lab   = pd.read_csv(d_labitems_path, compression="gzip", low_memory=False)

    def lower_cols(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.lower()
        return df

    # Normalize helpful columns to lowercase for robust matching
    d_items = lower_cols(d_items, ["label","abbreviation","category","unitname","linksto"])
    d_lab   = lower_cols(d_lab,   ["label","fluid","category","loinc_code"])
    # d_labitems uses "units" (not "unitname"); create a normalized view as "unitname" for uniformity
    if "units" in d_lab.columns and "unitname" not in d_lab.columns:
        d_lab = d_lab.rename(columns={"units": "unitname"})

    return d_items, d_lab

def prefer_metavision_first(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["priority_hint"] = (x["itemid"] < 220000).astype(int)  # 0 = MV (good), 1 = older CareVue
    return x.sort_values(["priority_hint", "itemid"])

def pick_chartevents(d_items: pd.DataFrame, patterns, max_n=8):
    """Select up to max_n chartevents itemids by regex over label/abbreviation."""
    if "linksto" in d_items.columns:
        ce_items = d_items[d_items["linksto"] == "chartevents"].copy()
    else:
        ce_items = d_items.copy()

    pat = re.compile("|".join(patterns), re.IGNORECASE)
    mask = pd.Series(False, index=ce_items.index)
    if "label" in ce_items.columns:
        mask = mask | ce_items["label"].astype(str).str.contains(pat, na=False)
    if "abbreviation" in ce_items.columns:
        mask = mask | ce_items["abbreviation"].astype(str).str.contains(pat, na=False)

    m = ce_items[mask].copy()
    if m.empty:
        return pd.DataFrame(columns=["itemid","unitname"])

    m = m.drop_duplicates(subset=["itemid"])
    m = prefer_metavision_first(m).head(max_n)
    # Ensure unit column exists (older dumps may miss it)
    if "unitname" not in m.columns:
        m["unitname"] = ""
    return m[["itemid","unitname"]]

def pick_labevents(d_lab: pd.DataFrame, labels_like, fluids=("blood","serum","plasma"), max_n=8):
    """Select up to max_n lab itemids by regex over label and restrict to fluids."""
    pat = re.compile("|".join(labels_like), re.IGNORECASE)
    mask = pd.Series(False, index=d_lab.index)
    if "label" in d_lab.columns:
        mask = mask | d_lab["label"].astype(str).str.contains(pat, na=False)

    m = d_lab[mask].copy()
    if "fluid" in m.columns:
        m = m[m["fluid"].isin(fluids)]
    if m.empty:
        return pd.DataFrame(columns=["itemid","unitname"])

    m = m.sort_values(["label","itemid"]).drop_duplicates(subset=["itemid"]).head(max_n)
    if "unitname" not in m.columns:
        m["unitname"] = ""
    return m[["itemid","unitname"]]

def build_varmap(d_items, d_lab, max_per_var=8) -> pd.DataFrame:
    """
    Define the 17 variables and select itemids for each.
    Regexes use non-capturing groups (?:...) to avoid Pandas warnings.
    """
    VAR_CFG = [
        # (variable, source, selector_fn, regex_patterns, to_unit)
        ("HeartRate",   "chartevents", pick_chartevents, [r"\bheart\s*rate\b", r"\bhr\b"], None),

        ("SysBP",       "chartevents", pick_chartevents,
            [r"(?:systolic).*(?:bp|blood\s*pressure)", r"\bsys\b", r"\bsystolic bp\b"], None),
        ("DiasBP",      "chartevents", pick_chartevents,
            [r"(?:diastolic).*(?:bp|blood\s*pressure)", r"\bdia\b", r"\bdiastolic bp\b"], None),
        ("MeanBP",      "chartevents", pick_chartevents,
            [r"(?:mean).*(?:bp|blood\s*pressure)", r"\bmap\b", r"\bmean arterial pressure\b"], None),

        ("RespRate",    "chartevents", pick_chartevents,
            [r"\bresp(?:iratory)?\s*rate\b", r"\brr\b"], None),

        ("Temperature", "chartevents", pick_chartevents,
            [r"\btemp(?:erature)?\b", r"\bcore temp\b", r"\boral temp\b", r"\brectal temp\b"], "c"),

        ("SpO2",        "chartevents", pick_chartevents,
            [r"\bspo2\b", r"oxygen\s*saturation", r"\bo2\s*sat"], None),

        # Labs (labevents)
        ("Sodium",      "labevents",   pick_labevents,  [r"\bsodium\b", r"\bna\b"], None),
        ("Potassium",   "labevents",   pick_labevents,  [r"\bpotassium\b", r"\bk\b"], None),
        ("Chloride",    "labevents",   pick_labevents,  [r"\bchloride\b", r"\bcl\b"], None),
        ("Bicarbonate", "labevents",   pick_labevents,  [r"\bbicarbonate\b", r"\btco2\b", r"\b(?:hco3|co2)\b"], None),
        ("BUN",         "labevents",   pick_labevents,  [r"\bbun\b", r"\burea\b", r"\burea nitrogen\b"], None),
        ("Creatinine",  "labevents",   pick_labevents,  [r"\bcreatinine\b"], None),
        ("Glucose",     "labevents",   pick_labevents,  [r"\bglucose\b"], None),
        ("Hematocrit",  "labevents",   pick_labevents,  [r"\bhemat(?:ocrit)?\b", r"\bhct\b"], None),
        ("WBC",         "labevents",   pick_labevents,  [r"\bwbc\b", r"white\s*blood\s*cells?"], None),
        ("Platelets",   "labevents",   pick_labevents,  [r"\bplate(?:let)?s?\b", r"\bplt\b"], None),
    ]

    rows = []
    for var, src, fn, pats, to_unit in VAR_CFG:
        hits = fn(d_items if src == "chartevents" else d_lab, pats, max_n=max_per_var)
        if hits.empty:
            print(f"[WARN] No matches for {var} ({src})", file=sys.stderr)
            continue

        hits = hits.dropna(subset=["itemid"]).copy()
        hits["itemid"] = hits["itemid"].astype(int)

        unit_series = hits["unitname"].where(hits["unitname"].notna(), "")
        for pri, (itemid, unitname) in enumerate(
            zip(hits["itemid"].tolist(), unit_series.astype(str).tolist()), start=1
        ):
            rows.append({
                "variable": var,
                "source": src,
                "itemid": int(itemid),
                "priority": int(pri),
                "unit": unitname or "",
                "to_unit": (to_unit or "").lower()
            })

    vm = pd.DataFrame(rows, columns=["variable","source","itemid","priority","unit","to_unit"])
    return vm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_items", default="d_items.csv.gz", help="Path to d_items.csv.gz")
    ap.add_argument("--d_labitems", default="d_labitems.csv.gz", help="Path to d_labitems.csv.gz")
    ap.add_argument("--out", default="varmap_mimiciv_17.csv", help="Output CSV path")
    ap.add_argument("--max_per_var", type=int, default=8, help="Max itemids per variable")
    args = ap.parse_args()

    d_items, d_lab = read_tables(args.d_items, args.d_labitems)
    vm = build_varmap(d_items, d_lab, max_per_var=args.max_per_var)

    if vm.empty:
        print("[ERROR] VarMap came out empty. Check your inputs.", file=sys.stderr)
        sys.exit(1)

    vm.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(vm)} rows across {vm['variable'].nunique()} variables.")
    # Quick preview
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(vm.groupby(["variable","source"])["itemid"].count().reset_index(name="n").to_string(index=False))

if __name__ == "__main__":
    main()
