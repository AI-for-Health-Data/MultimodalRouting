import pandas as pd
import gzip
import os

TSV_IMP    = "impressions_20250611.tsv"                  
COHORT_CSV = "INSPECT_full_dataset_with_impressions.csv" 
OUT_GZ     = "radiology_impressions_with_all_labels.csv.gz"

# 1. Load the raw impressions TSV
if not os.path.exists(TSV_IMP):
    raise FileNotFoundError(f"Missing impressions TSV: {TSV_IMP}")
print(f" Loading impressions from `{TSV_IMP}`…")
imp = pd.read_csv(
    TSV_IMP,
    sep="\t",
    dtype={"impression_id": int, "impressions": str},
    usecols=["impression_id", "impressions"],
    low_memory=False
)
imp = imp.rename(columns={"impressions": "impression_text"})
print(f"  → {len(imp):,} rows, columns = {imp.columns.tolist()}")

# 2. Load the cohort (one row per impression, with all your labels/demos)
if not os.path.exists(COHORT_CSV):
    raise FileNotFoundError(f"Missing cohort CSV: {COHORT_CSV}")
print(f"\n Loading cohort from `{COHORT_CSV}`…")
master = pd.read_csv(
    COHORT_CSV,
    dtype={"impression_id": int},
    low_memory=False
)
print(f"  → {len(master):,} rows, columns = {master.columns.tolist()}")

# 3. Drop any accidental duplicate impression_ids in the cohort
dups = master.impression_id.duplicated().sum()
if dups:
    print(f"  Found {dups:,} duplicated impression_id rows in cohort → dropping extras")
    master = master.drop_duplicates(subset=["impression_id"], keep="first")
print(f"  → now {len(master):,} unique‐keyed rows")

# 4. Merge – it should now be 1:1 on impression_id
print("\n Merging impressions → cohort (left join)…")
merged = master.merge(
    imp,
    on="impression_id",
    how="left",
    validate="one_to_one"
)
print(f"  → merged rows:             {len(merged):,}")
missing = merged["impression_text"].isna().sum()
print(f"  → missing impression_text: {missing:,} ({missing/len(merged):.1%})")

# 5. Peek at the first 5 non‐empty snippets
print("\nFirst 5 non‑empty `impression_text` snippets:")
for i, txt in enumerate(merged["impression_text"].dropna().astype(str).head(5), 1):
    snippet = txt.replace("\n"," ").strip()[:200]
    print(f" {i}. {snippet!r}")

# 6. Write out as a gzipped CSV
print(f"\n Writing → `{OUT_GZ}`…")
with gzip.open(OUT_GZ, "wt") as f:
    merged.to_csv(f, index=False)
print(" Done! Your file now has every radiology impression alongside all labels & demographics.")
