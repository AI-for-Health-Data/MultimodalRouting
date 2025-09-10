#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EHR cohort builder (MIMIC-IV) with:
  - In-hospital mortality label
  - COPD/bronchiectasis phenotype (ICD-9 list exactly as requested)
  - 17 time-series variables aggregated to 2-hour bins over the first 48 hours of ICU stay
  - Patient-level splits (train/val/test)
  - Normalization stats JSON (train-only mean, std, median per variable)
  - Optional CXR pairing (phenotyping: last CXR during stay; mortality: last CXR within first 48h)
  - Optional clinical notes aggregation and 512-token chunking (BioClinicalBERT-compatible)

Inputs (all in one folder, no subdirs):
  admissions.csv.gz
  patients.csv.gz
  diagnoses_icd.csv.gz
  labevents.csv.gz
  d_labitems.csv.gz
  icustays.csv.gz
  chartevents.csv.gz
  d_items.csv.gz
  varmap_mimiciv_17.csv   (variable map; multiple ITEMIDs allowed per variable)

Optional (for CXR pairing):
  mimic-cxr-2.0.0-metadata.csv.gz   (CXR metadata with StudyDate/StudyTime)
  [and if you want file paths] --cxr_files_root points to ".../mimic-cxr-jpg/2.0.0/files/"

Outputs (under --out_dir):
  ts_npz/stay_<stay_id>.npz     (X: 24xV imputed, M: 24xV mask, colnames, meta)
  cohort_master.csv             (one row per stay: labels, split, npz_path, optional dicom_id/path)
  normalization.json            (per-variable median, mean, std from TRAIN observed values only)
  notes_master.csv              (one row per subject with concatenated notes & BERT chunks) [if --notes_paths]
  cohort_with_notes.csv         (cohort master merged with per-subject chunk counts)        [if --notes_paths]
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------- Notes helpers ----------------------

def pick_note_time(df: pd.DataFrame) -> pd.Series:
    """Pick a usable timestamp column for notes."""
    for c in ["charttime", "note_time", "storetime", "chartdate", "chart_date"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                s.name = "note_time"
                return s
    raise ValueError("No usable timestamp in notes file (looked for charttime/note_time/storetime/chartdate).")

def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("\r\n","\n").replace("\r","\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def get_tokenizer():
    """Try BioClinicalBERT; fall back to whitespace tokenization."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=True)
        return ("hf", tok)
    except Exception:
        class WhiteSpaceTokenizer:
            def __init__(self):
                self.cls_token = "[CLS]"; self.sep_token = "[SEP]"
            def tokenize(self, text): return text.split()
        return ("ws", WhiteSpaceTokenizer())

def chunk_tokens(tokens, max_len): return [tokens[i:i+max_len] for i in range(0, len(tokens), max_len)]

def bio_bert_chunks(text: str, mode_tok, tokenizer, max_seq_len=512):
    """Return list of '[CLS] ... 512 ... [SEP]' strings."""
    if not text: return []
    content_len = max_seq_len - 2
    if mode_tok == "hf":
        toks = tokenizer.tokenize(text)
        pieces = chunk_tokens(toks, content_len)
        return [" ".join([tokenizer.cls_token] + p + [tokenizer.sep_token]) for p in pieces]
    else:
        toks = text.split()
        pieces = chunk_tokens(toks, content_len)
        return [" ".join(["[CLS]"] + p + ["[SEP]"]) for p in pieces]

# ---------------------- config / constants ----------------------

BIN_HOURS = 2           # 2-hour bins
WINDOW_HOURS = 48       # first 48h
N_BINS = WINDOW_HOURS // BIN_HOURS  # 24

# ---------------------- IO helpers ----------------------

def gzread(path, **kw):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, compression="gzip", **kw)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def j(*parts):
    return os.path.join(*parts)

# ---------------------- COPD ICD-9 helpers ----------------------

def _load_copd_icd9_list(path: str | None) -> List[str]:
    """
    Load ICD-9 code *roots* for COPD/bronchiectasis.
    If no file is given, defaults to 491, 492, 494, 496 (classic COPD+bronchiectasis set).
    Dots are ignored; we do prefix matching on the cleaned code roots.
    """
    if path and os.path.exists(path):
        codes = []
        # txt or csv: read first column / lines
        if path.lower().endswith((".txt", ".lst")):
            with open(path, "r") as f:
                for line in f:
                    c = re.sub(r"[^A-Za-z0-9]", "", line.strip().upper())
                    if c:
                        codes.append(c)
        else:
            df = pd.read_csv(path)
            if df.shape[1] == 0:
                return []
            col = df.columns[0]
            codes = (
                df[col].astype(str)
                .str.upper()
                .str.replace(".", "", regex=False)
                .str.replace(r"[^A-Za-z0-9]", "", regex=True)
                .tolist()
            )
        return sorted(set([c for c in codes if c]))
    # default roots
    return ["491", "492", "494", "496"]


def add_copd_label_icd9(diagnoses: pd.DataFrame, icd9_roots: List[str]) -> pd.DataFrame:
    """
    Return DataFrame(hadm_id, COPD_bronchiectasis) using ICD-9 *prefix* match on the provided roots.
    """
    if not icd9_roots:
        return pd.DataFrame({"hadm_id": [], "COPD_bronchiectasis": []})

    dx = diagnoses.copy()
    dx = dx[dx["icd_version"] == 9].copy()
    if dx.empty:
        return pd.DataFrame({"hadm_id": diagnoses["hadm_id"].unique(), "COPD_bronchiectasis": 0})

    dx["code"] = (
        dx["icd_code"].astype("string")
        .str.upper()
        .str.replace(".", "", regex=False)
        .str.strip()
    )

    # Build one regex that matches any root at the start
    roots = [re.escape(r) for r in icd9_roots]
    pat = re.compile(r"^(?:%s)" % "|".join(roots))

    hit = dx["code"].str.match(pat, na=False)
    pos = (
        hit.groupby(dx["hadm_id"])
        .any()
        .astype("int8")
        .rename("COPD_bronchiectasis")
        .reset_index()
    )
    return pos

# ---------------------- load core tables ----------------------

def load_core(data_dir: str):
    admissions = gzread(
        j(data_dir, "admissions.csv.gz"),
        usecols=["subject_id","hadm_id","admittime","dischtime","deathtime","hospital_expire_flag"],
        parse_dates=["admittime","dischtime","deathtime"],
        dtype={"subject_id":"int32","hadm_id":"int32","hospital_expire_flag":"int8"},
    )
    patients = gzread(
        j(data_dir, "patients.csv.gz"),
        usecols=["subject_id","anchor_age","anchor_year","gender"],
        dtype={"subject_id":"int32","anchor_age":"int16","anchor_year":"int16","gender":"category"},
    )
    icustays = gzread(
        j(data_dir, "icustays.csv.gz"),
        usecols=["subject_id","hadm_id","stay_id","intime","outtime"],
        parse_dates=["intime","outtime"],
        dtype={"subject_id":"int32","hadm_id":"int32","stay_id":"int32"},
    )
    diagnoses = gzread(
        j(data_dir, "diagnoses_icd.csv.gz"),
        usecols=["subject_id","hadm_id","icd_code","icd_version"],
        dtype={"subject_id":"int32","hadm_id":"int32","icd_code":"string","icd_version":"int8"},
    )
    # Metadata tables are loaded only to help verify items, not required downstream.
    d_items = gzread(j(data_dir, "d_items.csv.gz"))
    d_labitems = gzread(j(data_dir, "d_labitems.csv.gz"))
    return admissions, patients, icustays, diagnoses, d_items, d_labitems


# ---------------------- CXR metadata (optional) ----------------------

def load_cxr_metadata(cxr_metadata_path: str) -> pd.DataFrame:
    """
    Load MIMIC-CXR metadata and build a valid study datetime from DICOM fields.
    Handles messy StudyTime values by stripping non-digits, padding/truncating,
    and clamping HH to [0,23], MM/SS to [0,59].
    """
    meta = gzread(
        cxr_metadata_path,
        usecols=["subject_id","study_id","dicom_id","StudyDate","StudyTime","ViewPosition"],
        dtype={"subject_id":"int32","study_id":"int32","dicom_id":"string","ViewPosition":"category"},
    )

    def _safe_int(s: str, default: int = 0) -> int:
        try:
            return int(s)
        except Exception:
            return default

    def parse_study_dt(row):
        # --- StudyDate ---
        sd_raw = row.get("StudyDate")
        if pd.isna(sd_raw):
            return pd.NaT
        sd_digits = re.sub(r"\D", "", str(sd_raw))
        if len(sd_digits) < 8:
            return pd.NaT
        y, m, d = sd_digits[:4], sd_digits[4:6], sd_digits[6:8]

        # --- StudyTime (HHMMSS[.fraction]) ---
        st_raw = row.get("StudyTime")
        st_digits = re.sub(r"\D", "", "" if pd.isna(st_raw) else str(st_raw))
        # Pad/truncate to 6 digits (HHMMSS). Using rjust handles short forms like "930" -> "000930".
        st_digits = st_digits.rjust(6, "0")[:6]

        hh = _safe_int(st_digits[:2], 0)
        mm = _safe_int(st_digits[2:4], 0)
        ss = _safe_int(st_digits[4:6], 0)

        # Clamp to valid ranges to avoid pandas parser errors (e.g., "80:55:68")
        hh = min(max(hh, 0), 23)
        mm = min(max(mm, 0), 59)
        ss = min(max(ss, 0), 59)

        dt_str = f"{y}-{m}-{d} {hh:02d}:{mm:02d}:{ss:02d}"
        return pd.to_datetime(dt_str, errors="coerce")

    meta["study_datetime"] = meta.apply(parse_study_dt, axis=1)
    meta = meta.dropna(subset=["study_datetime"])
    return meta[["subject_id","study_id","dicom_id","study_datetime","ViewPosition"]]


def make_jpg_relpath(subject_id: int, study_id: int, dicom_id: str) -> str:
    """
    MIMIC-CXR-JPG v2 folder scheme:
      files/pXX/pXXXXXXXX/sYYYYYYYY/{dicom_id}.jpg
    where XX = first two digits of 8-digit subject_id (zero-padded).
    """
    sid = f"{int(subject_id):08d}"
    p2 = sid[:2]
    return os.path.join(f"p{p2}", f"p{sid}", f"s{int(study_id)}", f"{dicom_id}.jpg")

def link_cxr_to_stays(cxr_meta: pd.DataFrame,
                      admissions: pd.DataFrame,
                      icustays: pd.DataFrame) -> pd.DataFrame:
    # Join by subject, filter by admission window, then ICU stay window
    adm = admissions[["subject_id","hadm_id","admittime","dischtime"]]
    m = cxr_meta.merge(adm, on="subject_id", how="left")
    m = m[(m["study_datetime"] >= m["admittime"]) & (m["study_datetime"] <= m["dischtime"])].copy()

    stays = icustays[["subject_id","hadm_id","stay_id","intime","outtime"]]
    m = m.merge(stays, on=["subject_id","hadm_id"], how="left")
    m = m[(m["study_datetime"] >= m["intime"]) & (m["study_datetime"] <= m["outtime"])].copy()

    # If a dicom maps to multiple stays (rare), pick the one with study time closest to outtime
    m["dist_to_out"] = (m["outtime"] - m["study_datetime"]).abs()
    m = m.sort_values(["dicom_id","dist_to_out"]).drop_duplicates(subset=["dicom_id"], keep="first")
    return m[["subject_id","hadm_id","stay_id","study_id","dicom_id","study_datetime","ViewPosition"]]

# ---------------------- varmap ----------------------

def load_varmap(path_csv: str) -> pd.DataFrame:
    """
    Expected columns:
      variable,source,itemid,priority,unit,to_unit
    - source in {"chartevents","labevents"}
    - itemid is int
    - priority: lower number = higher priority
    """
    vm = pd.read_csv(path_csv, dtype={"variable":"string","source":"string","itemid":"int32",
                                      "priority":"int16","unit":"string","to_unit":"string"})
    vm["variable"] = vm["variable"].str.strip()
    vm["source"] = vm["source"].str.strip().str.lower()
    if not set(vm["source"].unique()) <= {"chartevents","labevents"}:
        raise ValueError("varmap source must be 'chartevents' or 'labevents'")
    return vm

# ---------------------- CCS mapping helpers ----------------------

def _load_ccs_keep_list(path_csv: str) -> Set[str]:
    """Load an optional set of CCS IDs to keep (e.g., the 25 MedFuse phenotypes)."""
    if not path_csv or not os.path.exists(path_csv):
        return set()
    df = pd.read_csv(path_csv)
    if df.shape[1] == 0:
        return set()
    col = df.columns[0]
    keep = set(df[col].astype(str).str.strip())
    return keep

def _load_ccs_maps(icd9_csv: str, icd10_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (map9, map10) with columns standardized to ['code','ccs_id','ccs_name'].
    """
    m9 = pd.DataFrame(columns=["code","ccs_id","ccs_name"])
    m10 = pd.DataFrame(columns=["code","ccs_id","ccs_name"])

    if icd9_csv and os.path.exists(icd9_csv):
        t9 = pd.read_csv(icd9_csv, dtype={"icd9":"string","ccs_id":"string","ccs_name":"string"})
        t9["code"] = t9["icd9"].str.upper().str.replace(".","", regex=False).str.strip()
        m9 = t9[["code","ccs_id","ccs_name"]].dropna()

    if icd10_csv and os.path.exists(icd10_csv):
        t10 = pd.read_csv(icd10_csv, dtype={"icd10":"string","ccs_id":"string","ccs_name":"string"})
        t10["code"] = t10["icd10"].str.upper().str.replace(".","", regex=False).str.strip()
        m10 = t10[["code","ccs_id","ccs_name"]].dropna()

    return m9, m10

def _ccs_multilabel_from_diagnoses(diagnoses: pd.DataFrame,
                                   map9: pd.DataFrame,
                                   map10: pd.DataFrame,
                                   keep_ids: Set[str]) -> pd.DataFrame:
    """
    Build multi-hot CCS labels per hadm_id.
    - diagnoses: columns ['hadm_id','icd_code','icd_version']
    - keep_ids: limit to these CCS IDs if non-empty; else keep all.
    Returns: DataFrame with one row per hadm_id and CCS_* columns (int8 {0,1})
    """
    dx = diagnoses.copy()
    dx["code"] = dx["icd_code"].astype("string").str.upper().str.replace(".","", regex=False).str.strip()
    dx["icd_version"] = dx["icd_version"].astype(int)

    dx9  = dx[dx["icd_version"] == 9].merge(map9, how="left", left_on="code", right_on="code")
    dx10 = dx[dx["icd_version"] == 10].merge(map10, how="left", left_on="code", right_on="code")
    m = pd.concat([dx9, dx10], ignore_index=True)

    # Drop unmapped rows
    m = m.dropna(subset=["ccs_id"])
    if m.empty:
        return pd.DataFrame(columns=["hadm_id"])  # will left-join, then fill with zeros

    m["ccs_id"] = m["ccs_id"].astype(str).str.strip()
    m["ccs_name"] = m["ccs_name"].astype(str).str.strip()

    if keep_ids:
        m = m[m["ccs_id"].isin(keep_ids)].copy()
        if m.empty:
            return pd.DataFrame(columns=["hadm_id"])

    # Construct stable column names like: CCS_108_COPD_and_bronchiectasis  (spaces & punctuation -> _)
    m["ccs_col"] = "CCS_" + m["ccs_id"].astype(str) + "_" + m["ccs_name"].str.replace(r"[^A-Za-z0-9]+","_", regex=True).str.strip("_")

    m["val"] = 1
    wide = (m[["hadm_id","ccs_col","val"]]
              .drop_duplicates()
              .pivot(index="hadm_id", columns="ccs_col", values="val")
              .fillna(0)
              .astype("int8")
              .reset_index())

    return wide

# ---------------------- labels & splits ----------------------

def compute_age_at_icu(icustays: pd.DataFrame, patients: pd.DataFrame) -> pd.Series:
    x = icustays.merge(patients, on="subject_id", how="left")
    def _age(r):
        base = 0 if pd.isna(r["anchor_age"]) else r["anchor_age"]
        return base + (r["intime"].year - r["anchor_year"])
    age = x.apply(_age, axis=1)
    return pd.Series(age, index=icustays.index).clip(lower=0)

def make_labels(icustays: pd.DataFrame,
                admissions: pd.DataFrame,
                patients: pd.DataFrame,
                diagnoses: pd.DataFrame,
                ccs_map9: pd.DataFrame,
                ccs_map10: pd.DataFrame,
                ccs_keep_ids: Set[str]) -> pd.DataFrame:
    """
    Builds the cohort table with:
      - age (metadata; NO age filter)
      - mortality (in-hospital, from admissions)
      - CCS multi-hot columns (hadm-level; optional restriction via ccs_keep_ids)
    """
    df = icustays.copy()
    df["age"] = compute_age_at_icu(df, patients)

    # In-hospital mortality (hadm-level)
    df = df.merge(admissions[["hadm_id","hospital_expire_flag"]], on="hadm_id", how="left")
    df = df.rename(columns={"hospital_expire_flag":"mortality"}).astype({"mortality":"int8"})

    # CCS labels (hadm-level)
    ccs_wide = _ccs_multilabel_from_diagnoses(diagnoses, ccs_map9, ccs_map10, ccs_keep_ids)
    df = df.merge(ccs_wide, on="hadm_id", how="left")

    # Fill missing CCS with zeros
    ccs_cols = [c for c in df.columns if c.startswith("CCS_")]
    for c in ccs_cols:
        df[c] = df[c].fillna(0).astype("int8")

    return df

def patient_level_split(subject_ids: pd.Series, seed: int = 2022,
                        train_frac: float = 0.7, val_frac: float = 0.1) -> Tuple[Set[int],Set[int],Set[int]]:
    sids = subject_ids.drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    n = len(sids)
    n_train = int(round(train_frac * n))
    n_val   = int(round(val_frac * n))
    train_ids = set(sids[:n_train])
    val_ids   = set(sids[n_train:n_train+n_val])
    test_ids  = set(sids[n_train+n_val:])
    return train_ids, val_ids, test_ids

def add_split_column(df: pd.DataFrame, tr: Set[int], va: Set[int], te: Set[int]) -> pd.DataFrame:
    def lab(x):
        if x in tr: return "train"
        if x in va: return "val"
        if x in te: return "test"
        return "other"
    df = df.copy()
    df["split"] = df["subject_id"].map(lab).astype("category")
    return df

def _parse_stay_id_series(s: pd.Series) -> pd.Series:
    """
    Try to coerce a variety of inputs to int stay_id.
    Handles strings like "stay_12345.npz" or "12345" etc.
    """
    def _one(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, np.integer)): return int(x)
        xs = str(x)
        m = re.search(r"(\d+)", xs)
        return int(m.group(1)) if m else np.nan
    out = s.map(_one)
    return out.astype("Int64")

def apply_predefined_splits(cohort: pd.DataFrame, listfiles_dir: str) -> pd.DataFrame:
    """
    Apply EXACT splits from MedFuse-style listfiles:
      train_listfile.csv, val_listfile.csv, test_listfile.csv
    Each must contain a stay identifier column (any of: 'stay_id', 'stay', 'icustay_id').
    The cohort is filtered to only those stays that appear in any of the three files.
    """
    def _load_one(fname: str) -> pd.DataFrame:
        p = os.path.join(listfiles_dir, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split file: {p}")
        df = pd.read_csv(p)
        # find a suitable stay column
        cand = [c for c in df.columns if c.lower() in {"stay_id","stay","icustay_id"}]
        if not cand:
            # try to fallback to first column if it looks like stay
            cand = [df.columns[0]]
        col = cand[0]
        df["stay_id"] = _parse_stay_id_series(df[col])
        df = df.dropna(subset=["stay_id"]).astype({"stay_id":"int64"})
        return df[["stay_id"]]

    tr = _load_one("train_listfile.csv"); tr["split"] = "train"
    va = _load_one("val_listfile.csv");   va["split"] = "val"
    te = _load_one("test_listfile.csv");  te["split"] = "test"

    all_df = pd.concat([tr, va, te], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["stay_id"], keep="first")

    # Filter cohort to these stays; assign split
    k = "stay_id"
    m = cohort.merge(all_df, on=k, how="inner")
    # Preserve category dtype
    m["split"] = m["split"].astype("category")

    # For transparency, ensure we didn't silently drop any duplicate stays with conflicting splits
    # (rare; if present in multiple files).
    # We kept first occurrence by drop_duplicates above.

    return m

# ---------------------- event fetch (chunked) ----------------------

def fetch_chartevents_subset(data_dir: str, itemids: List[int], stay_ids: List[int]) -> pd.DataFrame:
    usecols = ["stay_id","charttime","itemid","valuenum","valueuom"]
    ce_path = j(data_dir, "chartevents.csv.gz")
    chunks, keep = [], set(stay_ids)
    for chunk in pd.read_csv(ce_path, compression="gzip", usecols=usecols, parse_dates=["charttime"], chunksize=2_000_000):
        c = chunk[(chunk["itemid"].isin(itemids)) & (chunk["stay_id"].isin(keep))]
        if not c.empty: chunks.append(c)
    if chunks:
        out = pd.concat(chunks, ignore_index=True)
        return out.dropna(subset=["charttime","valuenum"])
    return pd.DataFrame(columns=usecols)

def fetch_labevents_subset(data_dir: str, itemids: List[int], hadm_ids: List[int]) -> pd.DataFrame:
    usecols = ["hadm_id","charttime","itemid","valuenum","valueuom"]
    le_path = j(data_dir, "labevents.csv.gz")
    chunks, keep = [], set(hadm_ids)
    for chunk in pd.read_csv(le_path, compression="gzip", usecols=usecols, parse_dates=["charttime"], chunksize=2_000_000):
        c = chunk[(chunk["itemid"].isin(itemids)) & (chunk["hadm_id"].isin(keep))]
        if not c.empty: chunks.append(c)
    if chunks:
        out = pd.concat(chunks, ignore_index=True)
        return out.dropna(subset=["charttime","valuenum"])
    return pd.DataFrame(columns=usecols)


# ---------------------- unit normalization ----------------------

def f_to_c(v): return (v - 32.0) * (5.0/9.0)

def normalize_value(variable: str, val: float, unit: str, to_unit: str) -> float:
    if pd.isna(val): return np.nan
    unit = (unit or "").strip().lower()
    to_unit = (to_unit or "").strip().lower()
    if variable == "Temperature":
        if unit in {"f","fahrenheit"} and to_unit in {"c","celsius"}:
            return f_to_c(float(val))
    return float(val)


# ---------------------- binning ----------------------

def make_bins(intime: pd.Timestamp) -> pd.IntervalIndex:
    edges = [intime + pd.Timedelta(hours=h) for h in range(0, WINDOW_HOURS + BIN_HOURS, BIN_HOURS)]
    return pd.IntervalIndex.from_breaks(edges, closed="right")

def assign_bin(t: pd.Timestamp, bins: pd.IntervalIndex) -> int:
    idx = bins.get_indexer([t])[0]
    return int(idx) if idx != -1 else -1

# ---------------------- per-stay aggregation ----------------------
def build_stay_matrix(stay_row, varmap, ce_df, le_df):
    intime = stay_row["intime"]
    bins = make_bins(intime)  # 24 bins

    variables = list(varmap["variable"].drop_duplicates())
    var_idx = {v: i for i, v in enumerate(variables)}
    V = len(variables)

    X = np.full((N_BINS, V), np.nan, dtype="float32")
    M = np.zeros((N_BINS, V), dtype="int8")

    stay_id = int(stay_row["stay_id"])
    hadm_id = int(stay_row["hadm_id"])

    ce = ce_df[ce_df["stay_id"] == stay_id] if not ce_df.empty else ce_df
    le = le_df[le_df["hadm_id"] == hadm_id] if not le_df.empty else le_df

    # NOTE: keys are (itemid, variable)
    item_priority = {(int(r.itemid), str(r.variable)): int(r.priority) for _, r in varmap.iterrows()}
    item_units = {
        (int(r.itemid), str(r.variable)): (
            "" if pd.isna(r.unit) else str(r.unit),
            "" if pd.isna(r.to_unit) else str(r.to_unit),
        )
        for _, r in varmap.iterrows()
    }

    # itemid -> [variables] (CE/LE compete)
    item2vars = varmap.groupby("itemid")["variable"].apply(list).to_dict()

    # single fused pool of candidates for both CE and LE
    candidates = [[[] for _ in range(V)] for _ in range(N_BINS)]

    def add_rows(df):
        if df.empty: return
        df = df.dropna(subset=["charttime", "valuenum"]).copy()
        df["bin_idx"] = df["charttime"].apply(lambda t: assign_bin(t, bins))
        df = df[(df["bin_idx"] >= 0) & (df["bin_idx"] < N_BINS)]
        for _, r in df.iterrows():
            itemid = int(r["itemid"])
            b = int(r["bin_idx"])
            vlist = item2vars.get(itemid, [])
            if not vlist:
                continue
            val = float(r["valuenum"])
            uom = r.get("valueuom")
            uom = "" if pd.isna(uom) else str(uom)
            for var in vlist:
                pr = item_priority.get((itemid, var), 999)
                unit, to_unit = item_units.get((itemid, var), ("", ""))
                valn = normalize_value(var, val, uom if uom else unit, to_unit if to_unit else unit)
                j = var_idx[var]
                candidates[b][j].append((pr, r["charttime"], valn))

    # Add CE and LE into the same candidate pool
    add_rows(ce)
    add_rows(le)

    # Pick lowest priority; among ties, latest time
    for b in range(N_BINS):
        for j in range(V):
            cand = candidates[b][j]
            if not cand:
                continue
            min_pr = min(p for p, _, _ in cand)
            pool = [c for c in cand if c[0] == min_pr]
            chosen = max(pool, key=lambda x: x[1] or pd.Timestamp.min)
            X[b, j] = float(chosen[2])
            M[b, j] = 1

    # Forward-fill within the 24 bins
    for j in range(V):
        last = np.nan
        for b in range(N_BINS):
            if not np.isnan(X[b, j]):
                last = X[b, j]
            else:
                X[b, j] = last if not np.isnan(last) else np.nan

    return X, M

def _safe_group_apply(gb, func):
    try:
        # pandas ≥ 2.2
        return gb.apply(func, include_groups=False)
    except TypeError:
        # pandas < 2.2
        return gb.apply(func)

# ---------------------- main ----------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EHR cohort builder (MIMIC-IV) aligned with MedFuse in all key details, plus optional chunked notes.

Features:
  - In-hospital mortality label (admissions.hospital_expire_flag)
  - Optional CCS phenotyping labels (single-level CCS; supports both ICD-9 and ICD-10 maps)
  - 17 time-series variables aggregated to 2-hour bins over the first 48 hours of ICU stay
  - Patient-level splits (train/val/test) OR exact split assignment from MedFuse-style listfiles
  - TRAIN-only normalization stats JSON (mean, std, median per variable)
  - Optional CXR pairing (phenotyping: LAST CXR during stay; mortality: LAST CXR within first 48h)
      • No view restriction and no view-based priority (choose last by time only), per MedFuse
  - Optional clinical notes aggregation and 512-token chunking (BioClinicalBERT-compatible)

Inputs (all in one folder, no subdirs unless otherwise noted):
  Required:
    admissions.csv.gz
    patients.csv.gz
    diagnoses_icd.csv.gz
    labevents.csv.gz
    d_labitems.csv.gz
    icustays.csv.gz
    chartevents.csv.gz
    d_items.csv.gz
    varmap_mimiciv_17.csv   (variable map; multiple ITEMIDs allowed per variable)
  Optional (for CXR pairing):
    mimic-cxr-2.0.0-metadata.csv.gz   (CXR metadata with StudyDate/StudyTime)
    [and if you want file paths] --cxr_files_root points to ".../mimic-cxr-jpg/2.0.0/files/"
  Optional (for EXACT splits like MedFuse):
    --split_listfiles_dir containing:
      train_listfile.csv
      val_listfile.csv
      test_listfile.csv
    Each CSV must contain a stay identifier column (e.g., 'stay_id' or 'stay'), possibly strings.
    The code tries to parse integers robustly.
  Optional (for CCS phenotyping):
    --ccs_icd9_map_csv (columns: icd9, ccs_id, ccs_name)
    --ccs_icd10_map_csv (columns: icd10, ccs_id, ccs_name)
    --ccs_keep_list_csv (one column listing CCS IDs to keep; e.g., 25 MedFuse CCS groups)
  Optional (for notes):
    --notes_paths (paths to one or more notes CSV.GZ)
    Each notes file must contain: subject_id, text, and a timestamp column named one of:
      charttime, note_time, storetime, chartdate, chart_date

Outputs (under --out_dir):
  ts_npz/stay_<stay_id>.npz     (X: 24xV z-scored & imputed, M: 24xV mask, colnames, meta)
  cohort_master.csv.gz          (one row per stay: labels, split, npz_path, optional dicom_id/path)
  normalization.json            (per-variable median, mean, std from TRAIN observed values only)
  notes_master.csv.gz           (one row per subject with concatenated notes & BERT chunks) [if --notes_paths]
  cohort_with_notes.csv.gz      (cohort master merged with per-subject chunk counts)        [if --notes_paths]
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------- Notes helpers ----------------------

def pick_note_time(df: pd.DataFrame) -> pd.Series:
    """Pick a usable timestamp column for notes."""
    for c in ["charttime", "note_time", "storetime", "chartdate", "chart_date"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                s.name = "note_time"
                return s
    raise ValueError("No usable timestamp in notes file (looked for charttime/note_time/storetime/chartdate).")

def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("\r\n","\n").replace("\r","\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def get_tokenizer():
    """Try BioClinicalBERT; fall back to whitespace tokenization."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=True)
        return ("hf", tok)
    except Exception:
        class WhiteSpaceTokenizer:
            def __init__(self):
                self.cls_token = "[CLS]"; self.sep_token = "[SEP]"
            def tokenize(self, text): return text.split()
        return ("ws", WhiteSpaceTokenizer())

def chunk_tokens(tokens, max_len):
    return [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]


def bio_bert_chunks(text: str, mode_tok, tokenizer, max_seq_len=512):
    """Return list of '[CLS] ... 512 ... [SEP]' strings."""
    if not text: return []
    content_len = max_seq_len - 2
    if mode_tok == "hf":
        toks = tokenizer.tokenize(text)
        pieces = chunk_tokens(toks, content_len)
        return [" ".join([tokenizer.cls_token] + p + [tokenizer.sep_token]) for p in pieces]
    else:
        toks = text.split()
        pieces = chunk_tokens(toks, content_len)
        return [" ".join(["[CLS]"] + p + ["[SEP]"]) for p in pieces]

# ---------------------- config / constants ----------------------

BIN_HOURS = 2           # 2-hour bins
WINDOW_HOURS = 48       # first 48h
N_BINS = WINDOW_HOURS // BIN_HOURS  # 24

# ---------------------- IO helpers ----------------------

def gzread(path, **kw):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, compression="gzip", **kw)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def j(*parts):
    return os.path.join(*parts)

# ---------------------- load core tables ----------------------

def load_core(data_dir: str):
    admissions = gzread(
        j(data_dir, "admissions.csv.gz"),
        usecols=["subject_id","hadm_id","admittime","dischtime","deathtime","hospital_expire_flag"],
        parse_dates=["admittime","dischtime","deathtime"],
        dtype={"subject_id":"int32","hadm_id":"int32","hospital_expire_flag":"int8"},
    )
    patients = gzread(
        j(data_dir, "patients.csv.gz"),
        usecols=["subject_id","anchor_age","anchor_year","gender"],
        dtype={"subject_id":"int32","anchor_age":"int16","anchor_year":"int16","gender":"category"},
    )
    icustays = gzread(
        j(data_dir, "icustays.csv.gz"),
        usecols=["subject_id","hadm_id","stay_id","intime","outtime"],
        parse_dates=["intime","outtime"],
        dtype={"subject_id":"int32","hadm_id":"int32","stay_id":"int32"},
    )
    diagnoses = gzread(
        j(data_dir, "diagnoses_icd.csv.gz"),
        usecols=["subject_id","hadm_id","icd_code","icd_version"],
        dtype={"subject_id":"int32","hadm_id":"int32","icd_code":"string","icd_version":"int8"},
    )
    # Metadata tables are loaded only to help verify items, not required downstream.
    d_items = gzread(j(data_dir, "d_items.csv.gz"))
    d_labitems = gzread(j(data_dir, "d_labitems.csv.gz"))
    return admissions, patients, icustays, diagnoses, d_items, d_labitems


# ---------------------- CXR metadata (optional) ----------------------

def load_cxr_metadata(cxr_metadata_path: str) -> pd.DataFrame:
    """
    Load MIMIC-CXR metadata and build a valid study datetime from DICOM fields.
    Handles messy StudyTime values by stripping non-digits, padding/truncating,
    and clamping HH to [0,23], MM/SS to [0,59].
    """
    meta = gzread(
        cxr_metadata_path,
        usecols=["subject_id","study_id","dicom_id","StudyDate","StudyTime","ViewPosition"],
        dtype={"subject_id":"int32","study_id":"int32","dicom_id":"string","ViewPosition":"category"},
    )

    def _safe_int(s: str, default: int = 0) -> int:
        try:
            return int(s)
        except Exception:
            return default

    def parse_study_dt(row):
        # --- StudyDate ---
        sd_raw = row.get("StudyDate")
        if pd.isna(sd_raw):
            return pd.NaT
        sd_digits = re.sub(r"\D", "", str(sd_raw))
        if len(sd_digits) < 8:
            return pd.NaT
        y, m, d = sd_digits[:4], sd_digits[4:6], sd_digits[6:8]

        # --- StudyTime (HHMMSS[.fraction]) ---
        st_raw = row.get("StudyTime")
        st_digits = re.sub(r"\D", "", "" if pd.isna(st_raw) else str(st_raw))
        st_digits = st_digits.rjust(6, "0")[:6]  # HHMMSS

        hh = _safe_int(st_digits[:2], 0)
        mm = _safe_int(st_digits[2:4], 0)
        ss = _safe_int(st_digits[4:6], 0)

        hh = min(max(hh, 0), 23)
        mm = min(max(mm, 0), 59)
        ss = min(max(ss, 0), 59)

        dt_str = f"{y}-{m}-{d} {hh:02d}:{mm:02d}:{ss:02d}"
        return pd.to_datetime(dt_str, errors="coerce")

    meta["study_datetime"] = meta.apply(parse_study_dt, axis=1)
    meta = meta.dropna(subset=["study_datetime"])
    return meta[["subject_id","study_id","dicom_id","study_datetime","ViewPosition"]]


def make_jpg_relpath(subject_id: int, study_id: int, dicom_id: str) -> str:
    """
    MIMIC-CXR-JPG v2 folder scheme:
      files/pXX/pXXXXXXXX/sYYYYYYYY/{dicom_id}.jpg
    where XX = first two digits of 8-digit subject_id (zero-padded).
    """
    sid = f"{int(subject_id):08d}"
    p2 = sid[:2]
    return os.path.join(f"p{p2}", f"p{sid}", f"s{int(study_id)}", f"{dicom_id}.jpg")

def link_cxr_to_stays(cxr_meta: pd.DataFrame,
                      admissions: pd.DataFrame,
                      icustays: pd.DataFrame) -> pd.DataFrame:
    # Join by subject, filter by admission window, then ICU stay window
    adm = admissions[["subject_id","hadm_id","admittime","dischtime"]]
    m = cxr_meta.merge(adm, on="subject_id", how="left")
    m = m[(m["study_datetime"] >= m["admittime"]) & (m["study_datetime"] <= m["dischtime"])].copy()

    stays = icustays[["subject_id","hadm_id","stay_id","intime","outtime"]]
    m = m.merge(stays, on=["subject_id","hadm_id"], how="left")
    m = m[(m["study_datetime"] >= m["intime"]) & (m["study_datetime"] <= m["outtime"])].copy()

    # If a dicom maps to multiple stays (rare), pick the one with study time closest to outtime
    m["dist_to_out"] = (m["outtime"] - m["study_datetime"]).abs()
    m = m.sort_values(["dicom_id","dist_to_out"]).drop_duplicates(subset=["dicom_id"], keep="first")
    return m[["subject_id","hadm_id","stay_id","study_id","dicom_id","study_datetime","ViewPosition"]]

# ---------------------- varmap ----------------------

def load_varmap(path_csv: str) -> pd.DataFrame:
    """
    Expected columns:
      variable,source,itemid,priority,unit,to_unit
    - source in {"chartevents","labevents"}
    - itemid is int
    - priority: lower number = higher priority
    """
    vm = pd.read_csv(path_csv, dtype={"variable":"string","source":"string","itemid":"int32",
                                      "priority":"int16","unit":"string","to_unit":"string"})
    vm["variable"] = vm["variable"].str.strip()
    vm["source"] = vm["source"].str.strip().str.lower()
    if not set(vm["source"].unique()) <= {"chartevents","labevents"}:
        raise ValueError("varmap source must be 'chartevents' or 'labevents'")
    return vm

# ---------------------- CCS mapping helpers ----------------------

def _load_ccs_keep_list(path_csv: str) -> Set[str]:
    """Load an optional set of CCS IDs to keep (e.g., the 25 MedFuse phenotypes)."""
    if not path_csv or not os.path.exists(path_csv):
        return set()
    df = pd.read_csv(path_csv)
    if df.shape[1] == 0:
        return set()
    col = df.columns[0]
    keep = set(df[col].astype(str).str.strip())
    return keep

def _load_ccs_maps(icd9_csv: str, icd10_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (map9, map10) with columns standardized to ['code','ccs_id','ccs_name'].
    """
    m9 = pd.DataFrame(columns=["code","ccs_id","ccs_name"])
    m10 = pd.DataFrame(columns=["code","ccs_id","ccs_name"])

    if icd9_csv and os.path.exists(icd9_csv):
        t9 = pd.read_csv(icd9_csv, dtype={"icd9":"string","ccs_id":"string","ccs_name":"string"})
        t9["code"] = t9["icd9"].str.upper().str.replace(".","", regex=False).str.strip()
        m9 = t9[["code","ccs_id","ccs_name"]].dropna()

    if icd10_csv and os.path.exists(icd10_csv):
        t10 = pd.read_csv(icd10_csv, dtype={"icd10":"string","ccs_id":"string","ccs_name":"string"})
        t10["code"] = t10["icd10"].str.upper().str.replace(".","", regex=False).str.strip()
        m10 = t10[["code","ccs_id","ccs_name"]].dropna()

    return m9, m10

def _ccs_multilabel_from_diagnoses(diagnoses: pd.DataFrame,
                                   map9: pd.DataFrame,
                                   map10: pd.DataFrame,
                                   keep_ids: Set[str]) -> pd.DataFrame:
    """
    Build multi-hot CCS labels per hadm_id.
    - diagnoses: columns ['hadm_id','icd_code','icd_version']
    - keep_ids: limit to these CCS IDs if non-empty; else keep all.
    Returns: DataFrame with one row per hadm_id and CCS_* columns (int8 {0,1})
    """
    dx = diagnoses.copy()
    dx["code"] = dx["icd_code"].astype("string").str.upper().str.replace(".","", regex=False).str.strip()
    dx["icd_version"] = dx["icd_version"].astype(int)

    dx9  = dx[dx["icd_version"] == 9].merge(map9, how="left", left_on="code", right_on="code")
    dx10 = dx[dx["icd_version"] == 10].merge(map10, how="left", left_on="code", right_on="code")
    m = pd.concat([dx9, dx10], ignore_index=True)

    # Drop unmapped rows
    m = m.dropna(subset=["ccs_id"])
    if m.empty:
        return pd.DataFrame(columns=["hadm_id"])  # will left-join, then fill with zeros

    m["ccs_id"] = m["ccs_id"].astype(str).str.strip()
    m["ccs_name"] = m["ccs_name"].astype(str).str.strip()

    if keep_ids:
        m = m[m["ccs_id"].isin(keep_ids)].copy()
        if m.empty:
            return pd.DataFrame(columns=["hadm_id"])

    # Stable column names: CCS_108_COPD_and_bronchiectasis
    m["ccs_col"] = "CCS_" + m["ccs_id"].astype(str) + "_" + m["ccs_name"].str.replace(r"[^A-Za-z0-9]+","_", regex=True).str.strip("_")

    m["val"] = 1
    wide = (m[["hadm_id","ccs_col","val"]]
              .drop_duplicates()
              .pivot(index="hadm_id", columns="ccs_col", values="val")
              .fillna(0)
              .astype("int8")
              .reset_index())

    return wide

# ---------------------- splits ----------------------

def compute_age_at_icu(icustays: pd.DataFrame, patients: pd.DataFrame) -> pd.Series:
    x = icustays.merge(patients, on="subject_id", how="left")
    def _age(r):
        base = 0 if pd.isna(r["anchor_age"]) else r["anchor_age"]
        return base + (r["intime"].year - r["anchor_year"])
    age = x.apply(_age, axis=1)
    return pd.Series(age, index=icustays.index).clip(lower=0)

def make_labels(icustays: pd.DataFrame,
                admissions: pd.DataFrame,
                patients: pd.DataFrame,
                diagnoses: pd.DataFrame,
                ccs_map9: pd.DataFrame,
                ccs_map10: pd.DataFrame,
                ccs_keep_ids: Set[str]) -> pd.DataFrame:
    """
    Builds the cohort table with:
      - age (metadata; NO age filter)
      - mortality (in-hospital, from admissions)
      - CCS multi-hot columns (hadm-level; optional restriction via ccs_keep_ids)
    """
    df = icustays.copy()
    df["age"] = compute_age_at_icu(df, patients)

    # In-hospital mortality (hadm-level)
    df = df.merge(admissions[["hadm_id","hospital_expire_flag"]], on="hadm_id", how="left")
    df = df.rename(columns={"hospital_expire_flag":"mortality"}).astype({"mortality":"int8"})

    # CCS labels (hadm-level)
    ccs_wide = _ccs_multilabel_from_diagnoses(diagnoses, ccs_map9, ccs_map10, ccs_keep_ids)
    df = df.merge(ccs_wide, on="hadm_id", how="left")

    # Fill missing CCS with zeros
    ccs_cols = [c for c in df.columns if c.startswith("CCS_")]
    for c in ccs_cols:
        df[c] = df[c].fillna(0).astype("int8")

    return df


def patient_level_split(subject_ids: pd.Series, seed: int = 2022,
                        train_frac: float = 0.7, val_frac: float = 0.1) -> Tuple[Set[int],Set[int],Set[int]]:
    sids = subject_ids.drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    n = len(sids)
    n_train = int(round(train_frac * n))
    n_val   = int(round(val_frac * n))
    train_ids = set(sids[:n_train])
    val_ids   = set(sids[n_train:n_train+n_val])
    test_ids  = set(sids[n_train+n_val:])
    return train_ids, val_ids, test_ids

def add_split_column(df: pd.DataFrame, tr: Set[int], va: Set[int], te: Set[int]) -> pd.DataFrame:
    def lab(x):
        if x in tr: return "train"
        if x in va: return "val"
        if x in te: return "test"
        return "other"
    df = df.copy()
    df["split"] = df["subject_id"].map(lab).astype("category")
    return df


def _parse_stay_id_series(s: pd.Series) -> pd.Series:
    """
    Try to coerce a variety of inputs to int stay_id.
    Handles strings like "stay_12345.npz" or "12345" etc.
    """
    def _one(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, np.integer)): return int(x)
        xs = str(x)
        m = re.search(r"(\d+)", xs)
        return int(m.group(1)) if m else np.nan
    out = s.map(_one)
    return out.astype("Int64")

def apply_predefined_splits(cohort: pd.DataFrame, listfiles_dir: str) -> pd.DataFrame:
    """
    Apply EXACT splits from MedFuse-style listfiles:
      train_listfile.csv, val_listfile.csv, test_listfile.csv
    Each must contain a stay identifier column (any of: 'stay_id', 'stay', 'icustay_id').
    The cohort is filtered to only those stays that appear in any of the three files.
    """
    def _load_one(fname: str) -> pd.DataFrame:
        p = os.path.join(listfiles_dir, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split file: {p}")
        df = pd.read_csv(p)
        # find a suitable stay column
        cand = [c for c in df.columns if c.lower() in {"stay_id","stay","icustay_id"}]
        if not cand:
            # try fallback to first column if it looks like stay
            cand = [df.columns[0]]
        col = cand[0]
        df["stay_id"] = _parse_stay_id_series(df[col])
        df = df.dropna(subset=["stay_id"]).astype({"stay_id":"int64"})
        return df[["stay_id"]]

    tr = _load_one("train_listfile.csv"); tr["split"] = "train"
    va = _load_one("val_listfile.csv");   va["split"] = "val"
    te = _load_one("test_listfile.csv");  te["split"] = "test"

    all_df = pd.concat([tr, va, te], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["stay_id"], keep="first")

    # Filter cohort to these stays; assign split
    m = cohort.merge(all_df, on="stay_id", how="inner")
    m["split"] = m["split"].astype("category")
    return m

# ---------------------- event fetch (chunked) ----------------------

def fetch_chartevents_subset(data_dir: str, itemids: List[int], stay_ids: List[int]) -> pd.DataFrame:
    usecols = ["stay_id","charttime","itemid","valuenum","valueuom"]
    ce_path = j(data_dir, "chartevents.csv.gz")
    chunks, keep = [], set(stay_ids)
    for chunk in pd.read_csv(ce_path, compression="gzip", usecols=usecols, parse_dates=["charttime"], chunksize=2_000_000):
        c = chunk[(chunk["itemid"].isin(itemids)) & (chunk["stay_id"].isin(keep))]
        if not c.empty: chunks.append(c)
    if chunks:
        out = pd.concat(chunks, ignore_index=True)
        return out.dropna(subset=["charttime","valuenum"])
    return pd.DataFrame(columns=usecols)

def fetch_labevents_subset(data_dir: str, itemids: List[int], hadm_ids: List[int]) -> pd.DataFrame:
    usecols = ["hadm_id","charttime","itemid","valuenum","valueuom"]
    le_path = j(data_dir, "labevents.csv.gz")
    chunks, keep = [], set(hadm_ids)
    for chunk in pd.read_csv(le_path, compression="gzip", usecols=usecols, parse_dates=["charttime"], chunksize=2_000_000):
        c = chunk[(chunk["itemid"].isin(itemids)) & (chunk["hadm_id"].isin(keep))]
        if not c.empty: chunks.append(c)
    if chunks:
        out = pd.concat(chunks, ignore_index=True)
        return out.dropna(subset=["charttime","valuenum"])
    return pd.DataFrame(columns=usecols)

# ---------------------- unit normalization ----------------------

def f_to_c(v): return (v - 32.0) * (5.0/9.0)

def normalize_value(variable: str, val: float, unit: str, to_unit: str) -> float:
    if pd.isna(val): return np.nan
    unit = (unit or "").strip().lower()
    to_unit = (to_unit or "").strip().lower()
    if variable == "Temperature":
        if unit in {"f","fahrenheit"} and to_unit in {"c","celsius"}:
            return f_to_c(float(val))
    return float(val)

# ---------------------- binning ----------------------

def make_bins(intime: pd.Timestamp) -> pd.IntervalIndex:
    edges = [intime + pd.Timedelta(hours=h) for h in range(0, WINDOW_HOURS + BIN_HOURS, BIN_HOURS)]
    return pd.IntervalIndex.from_breaks(edges, closed="right")

def assign_bin(t: pd.Timestamp, bins: pd.IntervalIndex) -> int:
    idx = bins.get_indexer([t])[0]
    return int(idx) if idx != -1 else -1

# ---------------------- per-stay aggregation ----------------------

def build_stay_matrix(stay_row, varmap, ce_df, le_df):
    intime = stay_row["intime"]
    bins = make_bins(intime)  # 24 bins

    variables = list(varmap["variable"].drop_duplicates())
    var_idx = {v: i for i, v in enumerate(variables)}
    V = len(variables)

    X = np.full((N_BINS, V), np.nan, dtype="float32")
    M = np.zeros((N_BINS, V), dtype="int8")

    stay_id = int(stay_row["stay_id"])
    hadm_id = int(stay_row["hadm_id"])

    ce = ce_df[ce_df["stay_id"] == stay_id] if not ce_df.empty else ce_df
    le = le_df[le_df["hadm_id"] == hadm_id] if not le_df.empty else le_df

    # NOTE: keys are (itemid, variable)
    item_priority = {(int(r.itemid), str(r.variable)): int(r.priority) for _, r in varmap.iterrows()}
    item_units = {
        (int(r.itemid), str(r.variable)): (
            "" if pd.isna(r.unit) else str(r.unit),
            "" if pd.isna(r.to_unit) else str(r.to_unit),
        )
        for _, r in varmap.iterrows()
    }

    # itemid -> [variables] (CE/LE compete)
    item2vars = varmap.groupby("itemid")["variable"].apply(list).to_dict()

    # single fused pool of candidates for both CE and LE
    candidates = [[[] for _ in range(V)] for _ in range(N_BINS)]

    def add_rows(df):
        if df.empty: return
        df = df.dropna(subset=["charttime", "valuenum"]).copy()
        df["bin_idx"] = df["charttime"].apply(lambda t: assign_bin(t, bins))
        df = df[(df["bin_idx"] >= 0) & (df["bin_idx"] < N_BINS)]
        for _, r in df.iterrows():
            itemid = int(r["itemid"])
            b = int(r["bin_idx"])
            vlist = item2vars.get(itemid, [])
            if not vlist:
                continue
            val = float(r["valuenum"])
            uom = r.get("valueuom")
            uom = "" if pd.isna(uom) else str(uom)
            for var in vlist:
                pr = item_priority.get((itemid, var), 999)
                unit, to_unit = item_units.get((itemid, var), ("", ""))
                valn = normalize_value(var, val, uom if uom else unit, to_unit if to_unit else unit)
                j = var_idx[var]
                candidates[b][j].append((pr, r["charttime"], valn))

    # Add CE and LE into the same candidate pool
    add_rows(ce)
    add_rows(le)

    # Pick lowest priority; among ties, latest time
    for b in range(N_BINS):
        for j in range(V):
            cand = candidates[b][j]
            if not cand:
                continue
            min_pr = min(p for p, _, _ in cand)
            pool = [c for c in cand if c[0] == min_pr]
            chosen = max(pool, key=lambda x: x[1] or pd.Timestamp.min)
            X[b, j] = float(chosen[2])
            M[b, j] = 1

    # Forward-fill within the 24 bins
    for j in range(V):
        last = np.nan
        for b in range(N_BINS):
            if not np.isnan(X[b, j]):
                last = X[b, j]
            else:
                X[b, j] = last if not np.isnan(last) else np.nan

    return X, M


def _safe_group_apply(gb, func):
    try:
        # pandas ≥ 2.2
        return gb.apply(func, include_groups=False)
    except TypeError:
        # pandas < 2.2
        return gb.apply(func)

# ---------------------- One-hot helpers ----------------------

def _load_onehot_spec(path):
    if not path:
        return {}
    with open(path, "r") as f:
        spec = json.load(f)
    return spec or {}

def _expand_onehot_for_var(x_raw_col: np.ndarray, m_raw_col: np.ndarray, var_name: str, rule: dict):
    """
    x_raw_col: shape (T,), raw values for one variable across time bins (NaNs for missing)
    m_raw_col: shape (T,), {0,1} mask for the same
    Returns: (oh_mat [T x K], oh_mask [T x K], oh_names [K])
    """
    T = x_raw_col.shape[0]
    rtype = (rule.get("type") or "").lower()

    if rtype == "categorical":
        cats = rule.get("values", [])
        K = len(cats)
        oh = np.zeros((T, K), dtype="float32")
        for k, v in enumerate(cats):
            match = (m_raw_col == 1) & np.isfinite(x_raw_col) & (x_raw_col == v)
            oh[match, k] = 1.0
        oh_mask = np.repeat(m_raw_col.reshape(-1,1), K, axis=1).astype("int8")
        names = [f"{var_name}=={v}" for v in cats]
        return oh, oh_mask, names

    if rtype == "range_int":
        vmin, vmax = int(rule["min"]), int(rule["max"])
        K = vmax - vmin + 1
        oh = np.zeros((T, K), dtype="float32")
        xv = np.round(x_raw_col).astype("float32")
        for k, val in enumerate(range(vmin, vmax+1)):
            match = (m_raw_col == 1) & np.isfinite(x_raw_col) & (xv == float(val))
            oh[match, k] = 1.0
        oh_mask = np.repeat(m_raw_col.reshape(-1,1), K, axis=1).astype("int8")
        names = [f"{var_name}=={val}" for val in range(vmin, vmax+1)]
        return oh, oh_mask, names

    # Unknown rule type -> no expansion
    return None, None, []

# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with all CSV.GZ files (no subdirs)")
    ap.add_argument("--varmap_csv", required=True, help="Mapping CSV for the 17 variables")
    ap.add_argument("--out_dir", default="out_ehr")
    ap.add_argument("--split_seed", type=int, default=2022)

    # Optional: predefined EXACT splits (MedFuse-style)
    ap.add_argument("--split_listfiles_dir", default=None,
                    help="Directory with train_listfile.csv, val_listfile.csv, test_listfile.csv for exact splits.")

    # Optional CXR pairing
    ap.add_argument("--cxr_metadata", default=None, help="Path to mimic-cxr-2.0.0-metadata.csv.gz")
    ap.add_argument("--cxr_files_root", default=None, help="Path to MIMIC-CXR-JPG files/ to emit paired_image_path")
    ap.add_argument("--ap_only", action="store_true",
                    help="If set, keep only AP view CXRs when pairing.")
    ap.add_argument("--task", choices=["phenotyping", "in-hospital-mortality"],
                    default="phenotyping",
                    help="Which pairing to expose as *_selected (phenotyping=last-in-stay, IHM=last-in-48h).")


    # Cohort constraints (to mirror MedFuse/Harutyunyan)
    ap.add_argument("--adult_only", action="store_true", help="Keep age >= 18 only.")
    ap.add_argument("--first_icu_only", action="store_true", help="Keep first ICU stay per subject.")

    # CCS maps & keep list (optional)
    ap.add_argument("--ccs_icd9_map_csv", default=None,
                    help="CSV with columns: icd9, ccs_id, ccs_name (single-level CCS).")
    ap.add_argument("--ccs_icd10_map_csv", default=None,
                    help="CSV with columns: icd10, ccs_id, ccs_name (single-level CCS).")
    ap.add_argument("--ccs_keep_list_csv", default=None,
                    help="Optional CSV listing CCS IDs to keep (e.g., MedFuse 25). If omitted, keeps all CCS categories present.")

    ap.add_argument("--copd_icd9_list", default=None,
                    help="TXT/CSV with one ICD-9 root per line (dots optional). Defaults to 491,492,494,496.")
    ap.add_argument("--no_copd_label", action="store_true",
                    help="If set, do NOT compute the COPD/bronchiectasis label.")

    # Notes
    ap.add_argument("--notes_paths", nargs="+", default=None,
                    help="One or more MIMIC-IV notes CSV.GZ files (e.g., noteevents.csv.gz radiology.csv.gz discharge.csv.gz)")
    ap.add_argument("--notes_out_dir", default=None,
                    help="Output dir for notes CSVs (defaults to --out_dir if omitted)")
    ap.add_argument("--notes_max_chunks", type=int, default=50,
                    help="Max 512-token chunks per patient (extra truncated)")
    ap.add_argument("--notes_min_chars", type=int, default=1,
                    help="Drop note snippets shorter than this many characters before concat")

    # One-hot expansion spec (optional)
    ap.add_argument("--onehot_spec", default=None,
                    help=("JSON spec for categorical expansions. If set, final tensors contain "
                          "z-scored continuous vars + one-hot categorical dims."))

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    out_npz = j(args.out_dir, "ts_npz"); ensure_dir(out_npz)

    print("Loading core MIMIC-IV tables ...")
    admissions, patients, icustays, diagnoses, _, _ = load_core(args.data_dir)

    print("Preparing cohort with labels (mortality + optional CCS multi-hot) ...")
    ccs_keep = _load_ccs_keep_list(args.ccs_keep_list_csv)
    ccs9, ccs10 = _load_ccs_maps(args.ccs_icd9_map_csv, args.ccs_icd10_map_csv)
    cohort = make_labels(icustays, admissions, patients, diagnoses, ccs9, ccs10, ccs_keep)

    # COPD/bronchiectasis ICD-9 label (hadm-level)
    if not args.no_copd_label:
        copd_roots = _load_copd_icd9_list(args.copd_icd9_list)
        copd_df = add_copd_label_icd9(diagnoses, copd_roots)
        cohort = cohort.merge(copd_df, on="hadm_id", how="left")
        cohort["COPD_bronchiectasis"] = cohort["COPD_bronchiectasis"].fillna(0).astype("int8")

  
    # Optional cohort constraints to mirror MedFuse/Harutyunyan exactly
    if getattr(args, "adult_only", False):
        cohort = cohort[cohort["age"] >= 18].copy()

    if getattr(args, "first_icu_only", False):
        cohort = (
            cohort.sort_values(["subject_id", "intime"])
                  .drop_duplicates(subset=["subject_id"], keep="first")
                  .copy()
        )

    # At least 48 hours in ICU (EHR features window)
    cohort["stay_hours"] = (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600.0
    cohort = cohort[cohort["stay_hours"] >= 48].copy()

    # Splits
    if args.split_listfiles_dir:
        print("Applying EXACT predefined splits from listfiles ...")
        cohort = apply_predefined_splits(cohort, args.split_listfiles_dir)
    else:
        print("No predefined listfiles given; doing patient-level random split ...")
        tr_ids, va_ids, te_ids = patient_level_split(cohort["subject_id"], seed=args.split_seed)
        cohort = add_split_column(cohort, tr_ids, va_ids, te_ids)

    # Optional: CXR pairing
    if args.cxr_metadata and os.path.exists(args.cxr_metadata):
        print("Loading CXR metadata and linking to ICU stays ...")
        cxr_meta = load_cxr_metadata(args.cxr_metadata)

        # Optional AP-only filter BEFORE linking to stays
        if args.ap_only:
            cxr_meta = cxr_meta[cxr_meta["ViewPosition"] == "AP"].copy()

        cxr_links = link_cxr_to_stays(cxr_meta, admissions, icustays)

        # ---- Take the LAST CXR by time ----
        def _pick_last_by_time(df: pd.DataFrame) -> pd.Series:
            df = df.sort_values(by=["study_datetime", "study_id", "dicom_id"], ascending=[True, True, True])
            row = df.iloc[-1]
            return pd.Series({
                "dicom_id": row["dicom_id"],
                "study_id": row["study_id"],
                "study_datetime": row["study_datetime"],
            })

        # Last CXR anywhere in the stay (phenotyping)
        last_in_stay = (
            _safe_group_apply(cxr_links.groupby("stay_id", group_keys=False), _pick_last_by_time)
            .reset_index()
            .rename(columns={
                "dicom_id": "paired_dicom_id_instay",
                "study_id": "paired_study_id_instay",
                "study_datetime": "paired_time_instay",
            })
        )

        # Last CXR within the first 48h from ICU intime (IHM)
        cxr_48 = cxr_links.merge(cohort[["stay_id","intime"]], on="stay_id", how="inner")
        cxr_48 = cxr_48[
            (cxr_48["study_datetime"] >= cxr_48["intime"]) &
            (cxr_48["study_datetime"] <= cxr_48["intime"] + pd.Timedelta(hours=48))
        ].copy()
        last_in_48 = (
            _safe_group_apply(cxr_48.groupby("stay_id", group_keys=False), _pick_last_by_time)
            .reset_index()
            .rename(columns={
                "dicom_id": "paired_dicom_id_48h",
                "study_id": "paired_study_id_48h",
                "study_datetime": "paired_time_48h",
            })
        )

        # Merge selections back to cohort
        cohort = cohort.merge(
            last_in_stay[["stay_id","paired_dicom_id_instay","paired_study_id_instay","paired_time_instay"]],
            on="stay_id", how="left"
        ).merge(
            last_in_48[["stay_id","paired_dicom_id_48h","paired_study_id_48h","paired_time_48h"]],
            on="stay_id", how="left"
        )

        # Optional: emit JPG paths
        if args.cxr_files_root:
            def path_instay(r):
                sid = r["subject_id"]
                sdid = r["paired_study_id_instay"]
                did = r["paired_dicom_id_instay"]
                if pd.isna(sdid) or pd.isna(did): return None
                return j(args.cxr_files_root, make_jpg_relpath(sid, int(sdid), str(did)))

            def path_48(r):
                sid = r["subject_id"]
                sdid = r["paired_study_id_48h"]
                did = r["paired_dicom_id_48h"]
                if pd.isna(sdid) or pd.isna(did): return None
                return j(args.cxr_files_root, make_jpg_relpath(sid, int(sdid), str(did)))

            cohort["paired_image_path_instay"] = cohort.apply(path_instay, axis=1)
            cohort["paired_image_path_48h"]    = cohort.apply(path_48, axis=1)

        # Task-based selected pairing
        if ("paired_dicom_id_instay" in cohort.columns) or ("paired_dicom_id_48h" in cohort.columns):
            if args.task == "phenotyping":
                cohort["paired_dicom_id_selected"]  = cohort["paired_dicom_id_instay"]
                cohort["paired_study_id_selected"]  = cohort["paired_study_id_instay"]
                cohort["paired_time_selected"]      = cohort["paired_time_instay"]
                if "paired_image_path_instay" in cohort.columns:
                    cohort["paired_image_path_selected"] = cohort["paired_image_path_instay"]
            else:  # in-hospital-mortality
                cohort["paired_dicom_id_selected"]  = cohort["paired_dicom_id_48h"]
                cohort["paired_study_id_selected"]  = cohort["paired_study_id_48h"]
                cohort["paired_time_selected"]      = cohort["paired_time_48h"]
                if "paired_image_path_48h" in cohort.columns:
                    cohort["paired_image_path_selected"] = cohort["paired_image_path_48h"]

    # Load varmap and fetch CE/LE subsets
    varmap = load_varmap(args.varmap_csv)
    variables = list(varmap["variable"].drop_duplicates())
    V = len(variables)
    print(f"Variables ({V}): {variables}")

    ce_itemids = sorted(varmap.loc[varmap["source"]=="chartevents","itemid"].unique().tolist())
    le_itemids = sorted(varmap.loc[varmap["source"]=="labevents","itemid"].unique().tolist())

    print("Streaming chartevents subset ...")
    ce_df = fetch_chartevents_subset(args.data_dir, ce_itemids, cohort["stay_id"].astype(int).tolist())
    print(f"  CE rows kept: {len(ce_df):,}")

    print("Streaming labevents subset ...")
    le_df = fetch_labevents_subset(args.data_dir, le_itemids, cohort["hadm_id"].astype(int).tolist())
    print(f"  LE rows kept: {len(le_df):,}")

    # Build per-stay matrices; collect TRAIN-only stats (observed values pre-imputation)
    rows = []
    obs_vals_by_var_train = defaultdict(list)  # values for mean/std/median (TRAIN only)

    ccs_cols_in_cohort = [c for c in cohort.columns if c.startswith("CCS_")]

    for _, row in cohort.iterrows():
        X, M = build_stay_matrix(row, varmap, ce_df, le_df)

        # Gather TRAIN observed values only
        if row["split"] == "train":
            for vidx in range(V):
                obs = X[M[:, vidx] == 1, vidx]
                if obs.size:
                    obs_vals_by_var_train[vidx].extend(obs.tolist())

        npz_path = j(out_npz, f"stay_{int(row['stay_id'])}.npz")
        np.savez_compressed(
            npz_path,
            X=X.astype("float32"),         # raw for now (NaNs remain, to be handled later)
            M=M.astype("int8"),
            colnames=np.array(variables, dtype=object),
            meta=np.array({
                "subject_id": int(row["subject_id"]),
                "hadm_id": int(row["hadm_id"]),
                "stay_id": int(row["stay_id"]),
                "age": float(row["age"]),
                "mortality": int(row["mortality"]),
                "split": str(row["split"]),
            }, dtype=object)
        )

        rec = {
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
            "stay_id": int(row["stay_id"]),
            "age": float(row["age"]),
            "mortality": int(row["mortality"]),
            "split": str(row["split"]),
            "npz_path": npz_path,
            "intime": row["intime"],
        }

        # Add COPD/bronchiectasis if present
        if "COPD_bronchiectasis" in cohort.columns:
            rec["COPD_bronchiectasis"] = int(row.get("COPD_bronchiectasis", 0))


        # Optional CCS presence summary
        if ccs_cols_in_cohort:
            rec["ccs_any"] = int(row[ccs_cols_in_cohort].sum() > 0)

        # Optional CXR pairing fields (including selected)
        for c in ["paired_dicom_id_instay","paired_time_instay",
                  "paired_dicom_id_48h","paired_time_48h",
                  "paired_study_id_instay","paired_study_id_48h",
                  "paired_image_path_instay","paired_image_path_48h",
                  "paired_dicom_id_selected","paired_study_id_selected",
                  "paired_time_selected","paired_image_path_selected"]:
            if c in cohort.columns:
                rec[c] = row.get(c, np.nan)

        rows.append(rec)

    # Compute TRAIN-only normalization stats (continuous vars)
    stats = {"variables": variables, "train": {"median": {}, "mean": {}, "std": {}}}
    train_medians = np.full(V, np.nan, dtype="float32")
    train_means   = np.full(V, np.nan, dtype="float32")
    train_stds    = np.full(V, np.nan, dtype="float32")

    for vidx, name in enumerate(variables):
        vals = np.array(obs_vals_by_var_train[vidx], dtype="float32")
        if len(vals):
            train_medians[vidx] = np.median(vals)
            train_means[vidx]   = float(np.mean(vals))
            train_stds[vidx]    = float(np.std(vals, ddof=0))
        stats["train"]["median"][name] = None if np.isnan(train_medians[vidx]) else float(train_medians[vidx])
        stats["train"]["mean"][name]   = None if np.isnan(train_means[vidx])   else float(train_means[vidx])
        stats["train"]["std"][name]    = None if np.isnan(train_stds[vidx])    else float(train_stds[vidx])

    with open(j(args.out_dir, "normalization.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # -------- Second pass: normalize + optional one-hot expansion --------
    onehot_spec = _load_onehot_spec(args.onehot_spec)

    for r in rows:
        d = np.load(r["npz_path"], allow_pickle=True)
        X_raw = d["X"].astype("float32")   # raw values, NaNs for missing
        M_raw = d["M"].astype("int8")
        colnames = list(d["colnames"])
        V_local = len(colnames)

        # 1) z-score non-categorical channels; leave categoricals raw for OH
        X_cont = X_raw.copy()
        for vidx in range(V_local):
            name = colnames[vidx]
            if name in onehot_spec:
                continue  # will be replaced by one-hots
            mu = train_means[vidx]
            sd = train_stds[vidx]
            if np.isnan(sd) or sd <= 0:
                sd = 1.0
            obs_mask = ~np.isnan(X_cont[:, vidx])
            if not np.isnan(mu):
                X_cont[obs_mask, vidx] = (X_cont[obs_mask, vidx] - mu) / sd
            else:
                X_cont[obs_mask, vidx] = 0.0
            X_cont[~obs_mask, vidx] = 0.0

        # 2) build final blocks: one-hot for categoricals; z-scored for others
        X_blocks = []
        M_blocks = []
        names_out = []

        for vidx, name in enumerate(colnames):
            if name in onehot_spec:
                oh, ohm, oh_names = _expand_onehot_for_var(X_raw[:, vidx], M_raw[:, vidx], name, onehot_spec[name])
                if oh is not None:
                    X_blocks.append(oh)
                    M_blocks.append(ohm)
                    names_out.extend(oh_names)
                # do NOT include the original continuous channel for categoricals
            else:
                X_blocks.append(X_cont[:, [vidx]])
                M_blocks.append(M_raw[:, [vidx]])
                names_out.append(name)

        X_final = np.concatenate(X_blocks, axis=1).astype("float32") if X_blocks else X_cont
        M_final = np.concatenate(M_blocks, axis=1).astype("int8")    if M_blocks else M_raw

        np.savez_compressed(
            r["npz_path"],
            X=X_final,
            M=M_final,
            colnames=np.array(names_out, dtype=object),
            meta=d["meta"]
        )

    # Write master CSV
    master = pd.DataFrame(rows)
    master.to_csv(j(args.out_dir, "cohort_master.csv.gz"), index=False, compression="gzip")

    # ====================== NOTES AUGMENT (optional) ======================
    if args.notes_paths:
        notes_out_dir = args.notes_out_dir or args.out_dir
        ensure_dir(notes_out_dir)

        # anchor window: earliest ICU intime per patient in current cohort
        cohort_for_notes = cohort[["subject_id","intime"]].copy()
        cohort_for_notes["intime"] = pd.to_datetime(cohort_for_notes["intime"], errors="coerce")
        first_intime = cohort_for_notes.groupby("subject_id")["intime"].min().rename("first_intime").reset_index()
        first_intime["first_intime_plus_48h"] = first_intime["first_intime"] + pd.Timedelta(hours=48)

        # load notes (e.g., discharge/radiology)
        note_frames = []
        for path in args.notes_paths:
            df = gzread(path)
            if "subject_id" not in df.columns:
                raise ValueError(f"{path} has no subject_id")
            if "text" not in df.columns:
                for alt in ["note_text","notes","report","body"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt:"text"})
                        break
                if "text" not in df.columns:
                    raise ValueError(f"{path} has no 'text' column")
            ts = pick_note_time(df)
            df = df.assign(note_time=ts)[["subject_id","text","note_time"]].dropna(subset=["note_time","text"])
            note_frames.append(df)

        notes_all = pd.concat(note_frames, ignore_index=True) if note_frames else pd.DataFrame(columns=["subject_id","text","note_time"])

        # filter to 0–48h post first ICU intime, clean, min length
        notes = notes_all.merge(first_intime, on="subject_id", how="inner")
        in_win = (notes["note_time"] >= notes["first_intime"]) & (notes["note_time"] <= notes["first_intime_plus_48h"])
        notes = notes[in_win].copy()
        notes["text"] = notes["text"].astype(str).map(clean_text)
        if args.notes_min_chars > 1:
            notes = notes[notes["text"].str.len() >= args.notes_min_chars]

        # concat per patient
        notes = notes.sort_values(["subject_id","note_time"])
        agg = (notes.groupby("subject_id")["text"]
                     .apply(lambda x: "\n\n".join(x))
                     .reset_index()
                     .rename(columns={"text":"notes_0_48h"}))

        # tokenize & chunk at 512 for BioClinicalBERT
        mode_tok, tokenizer = get_tokenizer()
        chunk_rows = []
        for _, r in agg.iterrows():
            sid = int(r["subject_id"])
            chunks = bio_bert_chunks(r["notes_0_48h"], mode_tok, tokenizer, max_seq_len=512)[:args.notes_max_chunks]
            rec = {"subject_id": sid, "num_chunks": len(chunks), "notes_0_48h": r["notes_0_48h"]}
            for i, ch in enumerate(chunks, 1):
                rec[f"bert_chunk_{i:03d}"] = ch
            chunk_rows.append(rec)

        notes_master = pd.DataFrame(chunk_rows)
        notes_master_path = j(notes_out_dir, "notes_master.csv.gz")
        notes_master.to_csv(notes_master_path, index=False, compression="gzip")

        # merge onto cohort_master by subject_id
        merge_cols = [c for c in notes_master.columns if c != "notes_0_48h"]
        master_notes = master.merge(notes_master[merge_cols], on="subject_id", how="left")
        master_notes_path = j(notes_out_dir, "cohort_with_notes.csv.gz")
        master_notes.to_csv(master_notes_path, index=False, compression="gzip")

        print("\n=== Notes augment ===")
        print(f"Patients with >=1 note in 0-48h: {notes_master['subject_id'].nunique():,}")
        print(f"Saved: {notes_master_path}")
        print(f"Saved: {master_notes_path}")
    # =================== END NOTES AUGMENT =====================

    # Quick verification
    print("\n=== Summary ===")
    print(f"Stays written: {len(master):,}")
    # Mortality prevalence
    print("Label prevalence (overall):", master[["mortality"]].mean().round(3).to_dict())

    if "COPD_bronchiectasis" in master.columns:
        print(f"COPD/bronchiectasis prevalence: {master['COPD_bronchiectasis'].mean():.3f}")

    # Top-10 CCS prevalence (if present)
    ccs_cols = [c for c in cohort.columns if c.startswith("CCS_")]
    if ccs_cols:
        prev = cohort[ccs_cols].mean().sort_values(ascending=False).round(3)
        top10 = prev.head(10).to_dict()
        print("Top-10 CCS prevalence:", top10)

    # Split counts
    print("Split counts:", master["split"].value_counts().to_dict())

    # CXR pairing counts (if present)
    if "paired_dicom_id_instay" in master.columns or "paired_dicom_id_48h" in master.columns:
        paired_instay = master.get("paired_dicom_id_instay", pd.Series(dtype=object)).notna().sum()
        paired_48h    = master.get("paired_dicom_id_48h", pd.Series(dtype=object)).notna().sum()
        print(f"Paired (last-in-stay): {paired_instay:,} | Paired (last-in-48h): {paired_48h:,}")

    print(f"\nSaved: {j(args.out_dir, 'cohort_master.csv.gz')}")
    print(f"Saved: {j(args.out_dir, 'normalization.json')}")
    if args.notes_paths:
        print(f"Saved: {j(args.notes_out_dir or args.out_dir, 'notes_master.csv.gz')}")
        print(f"Saved: {j(args.notes_out_dir or args.out_dir, 'cohort_with_notes.csv.gz')}")
    print(f"NPZ dir: {out_npz}")

    # ---- MedFuse-style split table (partial vs paired) ----
    def _per_split(df):
        order = ["train", "val", "test"]
        return {s: int((df["split"] == s).sum()) for s in order}

    # Prefer the task-selected pairing column if present
    if "paired_dicom_id_selected" in master.columns:
        paired_col = "paired_dicom_id_selected"
    else:
        paired_col = "paired_dicom_id_instay" if args.task == "phenotyping" else "paired_dicom_id_48h"

    if paired_col in master.columns:
        partial_counts = _per_split(master)  # (EHR + CXR)_partial
        paired_counts  = _per_split(master[master[paired_col].notna()])  # (EHR + CXR)_paired

        print("\n=== MedFuse-style split table ===")
        print(f"(EHR + CXR)_partial  -> {partial_counts}")
        print(f"(EHR + CXR)_paired   -> {paired_counts}")

        # For IHM, also show Positive/Negative rows like the paper
        if args.task == "in-hospital-mortality":
            pos = master[master["mortality"] == 1]
            neg = master[master["mortality"] == 0]
            pos_paired = pos[pos[paired_col].notna()]
            neg_paired = neg[neg[paired_col].notna()]
            print(f"Positive (partial): {_per_split(pos)}")
            print(f"Negative (partial): {_per_split(neg)}")
            print(f"Positive (paired):  {_per_split(pos_paired)}")
            print(f"Negative (paired):  {_per_split(neg_paired)}")
    else:
        print("\n[INFO] No paired column found; skipping MedFuse-style split table.")

if __name__ == "__main__":
    main()