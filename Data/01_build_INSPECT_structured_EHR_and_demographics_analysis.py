import os
import pandas as pd

# 1. Load your full cohort (with impressions merged in)
COHORT_CSV = "INSPECT_full_dataset_with_impressions.csv"
if not os.path.exists(COHORT_CSV):
    raise FileNotFoundError(f"Missing {COHORT_CSV}")
print(f" Loading cohort from {COHORT_CSV}")
cohort = pd.read_csv(
    COHORT_CSV,
    dtype={"impression_id": str, "person_id": int},
    parse_dates=["study_time"],
    low_memory=False,
)
print(f"  → cohort rows: {len(cohort):,}, unique patients: {cohort.person_id.nunique():,}")
person_ids = set(cohort.person_id)

# 2. Load OMOP vocabulary for concept names
VOCAB_CSV = "concept.csv"
print(f"\n Loading vocabulary from {VOCAB_CSV}")
vocab = pd.read_csv(VOCAB_CSV, usecols=["concept_id", "concept_name"], low_memory=False)
vocab["concept_id"] = vocab.concept_id.astype(int)
print(f"  → vocab rows: {len(vocab):,}")

# 3. Helper to build each OMOP domain table
def build_domain(fn, concept_col, domain_name):
    print(f"\n Building domain '{domain_name}' from {fn}")
    sample = pd.read_csv(fn, nrows=0)
    # pick any column with 'date' or 'time'
    date_cols = [c for c in sample.columns if ("date" in c.lower() or "time" in c.lower())]
    if not date_cols:
        raise ValueError(f"No date/time columns found in {fn}")
    print(f"  → parsing dates from: {date_cols}")

    # read full, parsing those columns
    df = pd.read_csv(fn, parse_dates=date_cols, low_memory=False)
    before = len(df)
    df = df[df.person_id.isin(person_ids)]
    print(f"  → cohort filter: {before:,} → {len(df):,}")

    # build event_time as first non-null among date_cols
    df["event_time"] = df[date_cols].bfill(axis=1).iloc[:,0]
    df = df[["person_id", "event_time", concept_col]].copy()
    df["event_type"] = domain_name
    df = df.rename(columns={concept_col: "concept_id"})
    df["concept_id"] = df.concept_id.astype(int)

    # merge in names
    df = df.merge(vocab, on="concept_id", how="left")
    df = df.rename(columns={"concept_name": f"{domain_name}_concept_name"})
    print(f"  → {domain_name} rows: {len(df):,}")
    return df[["person_id","event_time","event_type","concept_id",f"{domain_name}_concept_name"]]

# 4. Build each domain
conds = build_domain("condition_occurrence.csv", "condition_concept_id", "condition")
procs = build_domain("procedure_occurrence.csv", "procedure_concept_id", "procedure")
meas  = build_domain("measurement.csv",       "measurement_concept_id", "measurement")
drugs = build_domain("drug_exposure.csv",     "drug_concept_id",        "drug")

# 5. One‐row per patient: demographics + labels
print("\n Extracting demo + labels per patient")
demo = (
    cohort[[
        "person_id",
        "year_of_birth","gender_name","race_name","ethnicity_name",
        "pe_positive_nlp",
        "1_month_mortality","6_month_mortality","12_month_mortality",
        "1_month_readmission","6_month_readmission","12_month_readmission",
        "12_month_PH"
    ]]
    .drop_duplicates("person_id")
    .reset_index(drop=True)
)
print(f"  → demo rows: {len(demo):,}")

# 6. Concatenate everything into one “long” EHR table
print("\n Concatenating all domains + demo/labels")
all_ehr = pd.concat([demo, conds, procs, meas, drugs], axis=0, ignore_index=True, sort=False)
print(f"  → total rows: {len(all_ehr):,}, unique patients: {all_ehr.person_id.nunique():,}")

OUT = "structured_ehr_all.csv.gz"
all_ehr.to_csv(OUT, index=False, compression="gzip")
print(f"\n✔ Wrote compressed CSV → {OUT}")
print(f"  Rows: {len(all_ehr):,}")
print(f"  Unique patients: {all_ehr.person_id.nunique():,}")

# 7. Load the structured EHR CSV
EHR_CSV = "structured_ehr_all.csv.gz"
print(f" Loading structured EHR from {EHR_CSV}")
ehr = pd.read_csv(
    EHR_CSV,
    compression="gzip",
    parse_dates=["event_time"],
    dtype={          
        "gender_name": str,
        "race_name": str,
        "ethnicity_name": str,
        "pe_positive_nlp": str,
        "1_month_mortality": str,
        "6_month_mortality": str,
        "12_month_mortality": str,
        "1_month_readmission": str,
        "6_month_readmission": str,
        "12_month_readmission": str,
        "12_month_PH": str,
    },
    low_memory=False
)
print(f"  → Shape: {ehr.shape}")
print(f"  → Columns: {ehr.columns.tolist()}")

# 8. Load cohort for study_time & split
COHORT_CSV = "INSPECT_public/cohort/inspect_cohort.csv"
print(f"\n Loading cohort from {COHORT_CSV}")
cohort = pd.read_csv(
    COHORT_CSV,
    parse_dates=["study_time"],
    dtype={"person_id": int},
    low_memory=False
)
print(f"  → Cohort rows: {len(cohort):,}, unique patients: {cohort.person_id.nunique():,}")

# we only need one row per patient for the demographics + labels
demo = ehr.drop_duplicates("person_id")[[
    "person_id",
    "year_of_birth","gender_name","race_name","ethnicity_name",
    "pe_positive_nlp",
    "1_month_mortality","6_month_mortality","12_month_mortality",
    "1_month_readmission","6_month_readmission","12_month_readmission",
    "12_month_PH"
]]

df = cohort[["person_id","study_time","split"]].merge(demo,
    on="person_id", how="inner"
)

# 9. Compute age and age_group
# Age calculation
df["age"] = df.study_time.dt.year - df.year_of_birth

# Filter out implausible ages (like age < 0 or very old)
df = df[(df.age >= 0) & (df.age <= 120)]

# Adjust bins to move age=89 into ">89"
bins  = [0, 18, 39, 69, 88, 120]
labels= ["0-18", "18-39", "39-69", "69-88", ">88"]
df["age_group"] = pd.cut(df.age, bins=bins, labels=labels, right=True)

# Remove 0–18
df = df[df.age_group != "0-18"]

# Debugging print
print("Max age:", df.age.max())
print("Age group nulls:", df.age_group.isna().sum())
print("Age group counts:\n", df.age_group.value_counts(dropna=False))



# 10. Table 2: Demographics
print("\n=== Table 2: Demographics ===")
demo_vars = {
    "Gender"   : "gender_name",
    "Age"      : "age_group",
    "Race"     : "race_name",
    "Ethnicity": "ethnicity_name"
}
splits = ["train","valid","test"]
def pct(cnt, tot): return f"{cnt:>5} ({cnt/ tot*100:4.1f}%)"

for name, col in demo_vars.items():
    print(f"\n-- {name} --")
    total = len(df)
    vc = df[col].value_counts(dropna=False)
    for lvl, cnt in vc.items():
        print(f"{lvl:>12} : {pct(cnt, total)}")
    print()
    for s in splits:
        sub = df[df.split==s]
        vc_s = sub[col].value_counts()
        print(f" {s.upper():>5} (n={len(sub):,}):",
              "  ".join(f"{lvl}:{pct(vc_s.get(lvl,0), len(sub))}" 
                        for lvl in vc_s.index))
    print()

# 11. Table 4: Outcomes by split
print("\n=== Table 4: Outcomes by split ===")
outcomes = {
    "PE"      : "pe_positive_nlp",
    "Mort 1 m": "1_month_mortality",
    "Mort 6 m": "6_month_mortality",
    "Mort 12 m": "12_month_mortality",
    "Read 1 m": "1_month_readmission",
    "Read 6 m": "6_month_readmission",
    "Read 12 m": "12_month_readmission",
    "PH 12 m" : "12_month_PH"
}

for label, col in outcomes.items():
    print(f"\n-- {label} ({col}) --")
    vc_all = df[col].value_counts(dropna=False)
    print(" ALL: " + "  ".join(f"{lvl}:{cnt}" for lvl,cnt in vc_all.items()))
    for s in splits:
        vc_s = df[df.split==s][col].value_counts(dropna=False)
        print(f" {s.upper():>5}: " + "  ".join(f"{lvl}:{vc_s.get(lvl,0)}"
                                              for lvl in vc_s.index))

