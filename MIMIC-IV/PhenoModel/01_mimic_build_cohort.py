# Build MedFuse-style cohort files for PHENOTYPING only from your paired CSVs.

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PHENO_CLASSES = [
    'Acute and unspecified renal failure', 'Acute cerebrovascular disease',
    'Acute myocardial infarction', 'Cardiac dysrhythmias', 'Chronic kidney disease',
    'Chronic obstructive pulmonary disease and bronchiectasis',
    'Complications of surgical procedures or medical care', 'Conduction disorders',
    'Congestive heart failure; nonhypertensive', 'Coronary atherosclerosis and other heart disease',
    'Diabetes mellitus with complications', 'Diabetes mellitus without complication',
    'Disorders of lipid metabolism', 'Essential hypertension', 'Fluid and electrolyte disorders',
    'Gastrointestinal hemorrhage', 'Hypertension with complications and secondary hypertension',
    'Other liver diseases', 'Other lower respiratory disease', 'Other upper respiratory disease',
    'Pleurisy; pneumothorax; pulmonary collapse',
    'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
    'Respiratory failure; insufficiency; arrest (adult)', 'Septicemia (except in labor)', 'Shock'
]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def norm(s: str) -> str:
    s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

PHENO_NORM_LOOKUP = {norm(x): x for x in PHENO_CLASSES}

def find_id_col(df: pd.DataFrame) -> str:
    """Pick a stay id column (prefers 'stay_id')."""
    candidates = ["stay_id", "icustay_id", "stay", "icustay"]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError("Could not find a stay ID column among: " + ", ".join(candidates))

def find_image_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["image_path", "cxr_path", "img_path", "path"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def find_pheno_cols(df: pd.DataFrame) -> List[str]:
    col_norm = {norm(c): c for c in df.columns}
    out = []
    for label_norm, _human in PHENO_NORM_LOOKUP.items():
        if label_norm in col_norm:
            out.append(col_norm[label_norm])
    return out

def coerce_binary(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0, upper=1).astype(int)

def discover_note_text_columns(df: pd.DataFrame) -> List[str]:
    blocked_exact = set([
        "subject_id","hadm_id","admittime","intime","outtime",
        "image_path","cxr_path","img_path","path",
    ])
    blocked_exact.update(find_pheno_cols(df))
    if any(c in df.columns for c in ["stay_id","icustay_id","stay","icustay"]):
        blocked_exact.add(find_id_col(df))

    noteish = []
    for c in df.columns:
        cn = norm(c)
        if c in blocked_exact:
            continue
        if df[c].dtype == object:
            sample = df[c].dropna()
            if len(sample) == 0:
                continue
            svals = sample.astype(str).head(5).tolist()
            avg_len = np.mean([len(x) for x in svals]) if svals else 0
            if ("note" in cn) or ("text" in cn) or ("chunk" in cn) or avg_len >= 20:
                noteish.append(c)
    return noteish

def longify_notes(df: pd.DataFrame, id_col: str, note_cols: List[str]) -> pd.DataFrame:
    if len(note_cols) == 0:
        return pd.DataFrame(columns=["stay_id","text"])
    keep = [id_col] + note_cols
    wide = df[keep].copy()
    long = wide.melt(id_vars=[id_col], value_vars=note_cols, var_name="src", value_name="text")
    long = long.drop(columns=["src"])
    long["text"] = long["text"].astype(str)
    long["text"] = long["text"].str.strip()
    long = long.replace({"text": {"nan": ""}})
    long = long[long["text"].str.len() > 0]
    long = long.dropna(subset=["text"])
    long = long.rename(columns={id_col: "stay_id"})
    long = long[["stay_id","text"]].reset_index(drop=True)
    return long

def multilabel_split(ids: np.ndarray, Y: np.ndarray, seed: int = 42,
                     val_frac: float = 0.15, test_frac: float = 0.15) -> Dict[str, List[int]]:
    assert len(ids) == Y.shape[0]
    rng = np.random.default_rng(seed)
    num_pos = Y.sum(axis=1)
    # bin num_pos into 0..K using quantiles to reduce extreme class skew
    if len(ids) >= 100:
        qs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        edges = np.quantile(num_pos, qs)
        # ensure strictly increasing
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.array([num_pos.min()-1, num_pos.max()])
    else:
        edges = np.array([num_pos.min()-1, num_pos.max()])
    bins = np.digitize(num_pos, edges[1:-1], right=True)

    train_idx, val_idx, test_idx = [], [], []
    for b in np.unique(bins):
        mask = (bins == b)
        idx_b = np.where(mask)[0]
        rng.shuffle(idx_b)
        nb = len(idx_b)
        n_test = int(round(test_frac * nb))
        n_val  = int(round(val_frac  * nb))
        n_train = nb - n_val - n_test
        sel_test = idx_b[:n_test]
        sel_val  = idx_b[n_test:n_test+n_val]
        sel_tr   = idx_b[n_test+n_val:]
        train_idx.extend(sel_tr.tolist())
        val_idx.extend(sel_val.tolist())
        test_idx.extend(sel_test.tolist())

    return {
        "train": ids[train_idx].tolist(),
        "val":   ids[val_idx].tolist(),
        "test":  ids[test_idx].tolist(),
    }

def main():
    ap = argparse.ArgumentParser(description="Build MedFuse-style cohort files for phenotyping.")
    ap.add_argument("--paired-dir", type=str, default="./paired_with_notes",
                    help="Directory containing paired CSVs you created.")
    ap.add_argument("--paired-csv", type=str, default=None,
                    help="Specific CSV to use (default: tries paired_phenotyping_with_note_chunks.csv then paired_phenotyping_with_note.csv).")
    ap.add_argument("--structured-path", type=str, default=None,
                    help="Path to an existing structured_24h.parquet to reuse (RECOMMENDED).")
    ap.add_argument("--out-dir", type=str, default="./data/MIMIC-IV",
                    help="Output directory for parquet files and splits.json.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # pick the paired CSV (notes + labels + possibly image paths) 
    if args.paired_csv is not None:
        paired_path = args.paired_csv
        if not os.path.isfile(paired_path):
            print(f"[ERROR] --paired-csv not found: {paired_path}", file=sys.stderr)
            sys.exit(1)
    else:
        candidates = [
            os.path.join(args.paired_dir, "paired_phenotyping_with_note_chunks.csv"),
            os.path.join(args.paired_dir, "paired_phenotyping_with_note.csv"),
            os.path.join(args.paired_dir, "paired_ehr_cxr_phenotyping.csv"),
        ]
        paired_path = next((p for p in candidates if os.path.isfile(p)), None)
        if paired_path is None:
            print("[ERROR] Could not find a phenotyping paired CSV in:", file=sys.stderr)
            for c in candidates:
                print("  -", c, file=sys.stderr)
            sys.exit(1)

    print(f"[info] Using paired phenotyping CSV: {paired_path}")
    base = pd.read_csv(paired_path, low_memory=False)

    # find the ID, label, image, note columns 
    id_col = find_id_col(base)
    ph_cols = find_pheno_cols(base)
    if len(ph_cols) != 25:
        print(f"[warn] Found {len(ph_cols)} phenotype columns (expected 25). Will keep what we found.")

    image_col = find_image_col(base)
    note_cols = discover_note_text_columns(base)

    # labels.parquet (stay_id + 25 binary columns)
    # Map label column order back to canonical PHENO_CLASSES order if present
    col_norm_rev = {norm(c): c for c in ph_cols}
    ordered_cols = [col_norm_rev[norm(lbl)] for lbl in PHENO_CLASSES if norm(lbl) in col_norm_rev]
    labels_df = base[[id_col] + ordered_cols].copy()
    labels_df = labels_df.rename(columns={id_col: "stay_id"})
    labels_df[ordered_cols] = coerce_binary(labels_df[ordered_cols])
    labels_out = os.path.join(args.out_dir, "labels.parquet")
    labels_df.to_parquet(labels_out, index=False)
    print(f"[ok] wrote {labels_out}  (n={len(labels_df):,}, labels={len(ordered_cols)})")

    # notes_24h.parquet (long: stay_id, text) 
    if len(note_cols) == 0:
        print("[warn] Did not find any note/text/chunk columns; writing empty notes_24h.parquet")
        notes_df = pd.DataFrame(columns=["stay_id","text"])
    else:
        notes_df = longify_notes(base, id_col=id_col, note_cols=note_cols)
    notes_out = os.path.join(args.out_dir, "notes_24h.parquet")
    notes_df.to_parquet(notes_out, index=False)
    print(f"[ok] wrote {notes_out}  (rows={len(notes_df):,} from {len(note_cols)} text cols)")

    # images_24h.parquet (long: stay_id, image_path) 
    if image_col is None:
        print("[warn] Did not find an image path column; writing empty images_24h.parquet")
        images_df = pd.DataFrame(columns=["stay_id","image_path"])
    else:
        images_df = base[[id_col, image_col]].dropna()
        images_df = images_df.rename(columns={id_col: "stay_id", image_col: "image_path"})
        images_df["image_path"] = images_df["image_path"].astype(str).str.strip()
        images_df = images_df[images_df["image_path"].str.len() > 0]
    images_out = os.path.join(args.out_dir, "images_24h.parquet")
    images_df.to_parquet(images_out, index=False)
    print(f"[ok] wrote {images_out}  (rows={len(images_df):,})")

    # structured_24h.parquet 
    if args.structured_path is None:
        default_struct = os.path.join(args.out_dir, "structured_24h.parquet")
        if os.path.isfile(default_struct):
            args.structured_path = default_struct
            print(f"[info] Reusing existing structured file: {default_struct}")
        else:
            print("[ERROR] structured_24h.parquet is required but not provided.\n"
                  "        Pass --structured-path pointing to your existing 24×F features (stay_id, hour, feat_*).",
                  file=sys.stderr)
            sys.exit(1)

    struct_src = args.structured_path
    struct_dst = os.path.join(args.out_dir, "structured_24h.parquet")
    if os.path.abspath(struct_src) != os.path.abspath(struct_dst):
        struct_df = pd.read_parquet(struct_src)
        if "stay_id" not in struct_df.columns or "hour" not in struct_df.columns:
            raise RuntimeError("structured_24h must contain columns: stay_id, hour, plus feature columns.")
        if struct_df.drop(columns=["stay_id","hour"]).shape[1] == 0:
            raise RuntimeError("structured_24h has no feature columns beyond stay_id/hour.")
        struct_df.to_parquet(struct_dst, index=False)
        print(f"[ok] wrote {struct_dst}  (rows={len(struct_df):,}, feats={struct_df.shape[1]-2})")
    else:
        struct_df = pd.read_parquet(struct_dst)
        print(f"[ok] found {struct_dst}  (rows={len(struct_df):,}, feats={struct_df.shape[1]-2})")


    lab = labels_df.copy()
    struct_ids = pd.read_parquet(struct_dst, columns=["stay_id"])["stay_id"].unique()
    lab = lab[lab["stay_id"].isin(struct_ids)].reset_index(drop=True)

    Y = lab[ordered_cols].values
    ids = lab["stay_id"].values
    splits = multilabel_split(ids, Y, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac)
    splits_out = os.path.join(args.out_dir, "splits.json")
    with open(splits_out, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[ok] wrote {splits_out}  (train={len(splits['train']):,}, val={len(splits['val']):,}, test={len(splits['test']):,})")

    n = len(lab)
    pos_counts = Y.sum(axis=0)
    print("\n[summary] Phenotyping cohort")
    print(f"  stays with labels & structured: {n:,}")
    for lbl, pos in zip(ordered_cols, pos_counts):
        print(f"    - {lbl}: pos={int(pos):,}  prev={pos/n:0.2%}")

if __name__ == "__main__":
    main()
