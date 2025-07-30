import os, sys, glob
import pandas as pd

for d in ("INSPECT_public/cohort", "INSPECT_public/reports", "filtered_ehr"):
    os.makedirs(d, exist_ok=True)

META_TSV    = "study_metadata_20250611.tsv"   
MAP_TSV     = "study_mapping_20250611.tsv"    
LABELS_TSV  = "labels_20250611.tsv"
SPLITS_TSV  = "splits_20250611.tsv"
IMPRESS_CSV = "Final_Impressions.csv"
VOCAB_CSV   = "concept.csv"

for f in (META_TSV, MAP_TSV, LABELS_TSV, SPLITS_TSV, IMPRESS_CSV, VOCAB_CSV):
    if not os.path.exists(f):
        sys.exit(f" Required file missing: {f}")

# 1. Load imaging metadata (DICOM header CSV keyed by impression_id)
df_meta = pd.read_csv(META_TSV, sep="\t", low_memory=False)
if "impression_id" not in df_meta.columns:
    sys.exit(" `impression_id` not in " + META_TSV)
print(f" Loaded imaging metadata: {df_meta.shape}")

# 2. Load mapping → brings in person_id + procedure_DATETIME per impression
df_map = pd.read_csv(MAP_TSV, sep="\t", low_memory=False)
for c in ("impression_id","person_id","procedure_DATETIME"):
    if c not in df_map.columns:
        sys.exit(f" `{c}` missing from {MAP_TSV}")

df_map = (
    df_map
    .rename(columns={"procedure_DATETIME":"study_time"})
    .assign(study_time=lambda d: pd.to_datetime(d["study_time"], errors="coerce"))
    .loc[:, ["impression_id","person_id","study_time"]]
    .drop_duplicates(subset=["impression_id"])
)
print(f" Loaded mapping: {df_map.shape}")

# 3. Build the cohort: merge metadata ↔ mapping
df_cohort = df_meta.merge(df_map, on="impression_id", how="inner")
print(f" Merged metadata ↔ mapping: {df_cohort.shape}")

# 4. Merge in labels (keyed by impression_id), dropping any stray person_id
df_lbl = (
    pd.read_csv(LABELS_TSV, sep="\t", low_memory=False)
    .drop(columns=["person_id"], errors="ignore")
)
if "impression_id" not in df_lbl.columns:
    sys.exit(f" `impression_id` missing from {LABELS_TSV}")
df_cohort = df_cohort.merge(df_lbl, on="impression_id", how="left")
print(f" Merged labels: {df_cohort.shape}")

# 5. Merge in splits (keyed by impression_id), dropping stray person_id
df_split = pd.read_csv(SPLITS_TSV, sep="\t", low_memory=False)
df_split = df_split.drop(columns=["person_id"], errors="ignore")
if "impression_id" not in df_split.columns:
    sys.exit(f" `impression_id` missing from {SPLITS_TSV}")
# if the split‐column has a weird name, rename it to “split”
if "split" not in df_split.columns:
    alt = [c for c in df_split.columns if "split" in c.lower()]
    if not alt:
        sys.exit(f" could not find any `split` column in {SPLITS_TSV}")
    df_split = df_split.rename(columns={alt[0]:"split"})
df_cohort = df_cohort.merge(df_split[["impression_id","split"]], on="impression_id", how="left")
print(f" Merged splits: {df_cohort.shape}")

# 6. Write the clean master cohort
cohort_csv = "INSPECT_public/cohort/inspect_cohort.csv"
df_cohort.to_csv(cohort_csv, index=False)
print(f" Wrote cohort: {df_cohort.shape} → {cohort_csv}")

# 7. Copy the impressions file into reports/
pd.read_csv(IMPRESS_CSV, low_memory=False)\
  .to_csv("INSPECT_public/reports/inspect_impressions.csv", index=False)
print(" Copied impressions → INSPECT_public/reports/inspect_impressions.csv")

# 8. Filter & decode each OMOP CSV on person_id ≤ study_time
cohort_df = pd.read_csv(cohort_csv, parse_dates=["study_time"], low_memory=False)
vocab     = pd.read_csv(VOCAB_CSV, usecols=["concept_id","concept_name"], low_memory=False)
SKIP      = {META_TSV, MAP_TSV, LABELS_TSV, SPLITS_TSV, IMPRESS_CSV, VOCAB_CSV, os.path.basename(__file__)}

for tbl in glob.glob("*.csv"):
    if tbl in SKIP:
        continue
    df = pd.read_csv(tbl, low_memory=False)
    if "person_id" not in df.columns:
        print(f"– skipping {tbl} (no person_id)"); continue

    # pick the first date/time column
    times = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if not times:
        print(f"– skipping {tbl} (no time/date column)"); continue
    tcol = times[0]
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")

    # join to get study_time, then filter
    merged = df.merge(cohort_df[["person_id","study_time"]], on="person_id", how="inner")
    before, after = len(merged), len(merged[merged[tcol] <= merged["study_time"]])
    merged = merged[merged[tcol] <= merged["study_time"]]

    # decode any *_concept_id columns
    for c in [c for c in merged.columns if c.endswith("_concept_id")]:
        merged = merged.merge(
            vocab.rename(columns={
                "concept_id": c,
                "concept_name": c.replace("_concept_id","") + "_name"
            }),
            on=c, how="left"
        )

    out = f"filtered_ehr/filtered_{tbl}"
    if tbl in ("measurement.csv", "observation.csv", "condition_occurrence.csv"):
        out = out.replace(".csv", ".parquet")
        merged.to_parquet(out, index=False)
    else:
        merged.to_csv(out, index=False)

    print(f" {tbl}: {before}→{after} rows → {out}")

# 9. Print Table 4’s eight label counts as a final sanity check
print("\n=== Table 4 label counts ===")
tasks = {
    "PE"       : "pe_positive_nlp",
    "Mort 1 m" : "1_month_mortality",
    "Mort 6 m" : "6_month_mortality",
    "Mort 12 m": "12_month_mortality",
    "Read 1 m" : "1_month_readmission",
    "Read 6 m" : "6_month_readmission",
    "Read 12 m": "12_month_readmission",
    "PH 12 m"  : "12_month_PH"
}
for name, col in tasks.items():
    if col in df_cohort:
        print(f"\n--- {name} ({col}) ---")
        vc = df_cohort[col].value_counts(dropna=False).sort_index()
        for val, cnt in vc.items():
            print(f"  {str(val):>8} : {cnt}")

print("\n All done!")

# 10. Merge cohort + impressions → Full dataset with labels + text
print("\n Merging cohort with impressions...")

cohort = pd.read_csv("INSPECT_public/cohort/inspect_cohort.csv", low_memory=False)
impr   = pd.read_csv("INSPECT_public/reports/inspect_impressions.csv", low_memory=False)

# Merge on impression_id
full = cohort.merge(impr, on="impression_id", how="left")

# Save as compressed CSV
full.to_csv("INSPECT_full_dataset_with_impressions.csv.gz", index=False, compression="gzip")

print(f" Wrote compressed full dataset → INSPECT_full_dataset_with_impressions.csv.gz")
print(f"  Rows: {len(full):,}")
print(f"  Unique patients: {full.person_id.nunique():,}")
