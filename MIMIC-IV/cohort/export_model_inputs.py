import os, json, argparse, re
import numpy as np
import pandas as pd

def read_master(cohort_dir: str) -> pd.DataFrame:
    path = os.path.join(cohort_dir, "cohort_master.csv.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path, compression="gzip")
    # Ensure required keys exist
    need = {"stay_id","subject_id","split","mortality","npz_path"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df

def load_npz_safely(p):
    try:
        return np.load(p, allow_pickle=True)
    except Exception:
        return None

def build_structured(master: pd.DataFrame, out_path: str):
    rows = []
    feat_names = None
    n_bad = 0

    for _, r in master.iterrows():
        p = r["npz_path"]
        d = load_npz_safely(p)
        if d is None:
            n_bad += 1
            continue
        X = d["X"].astype("float32")   
        cols = list(d["colnames"])
        T, F = X.shape
        if feat_names is None:
            feat_names = cols
        elif len(cols) != len(feat_names):
            raise ValueError(f"Feature dim mismatch at {p}: {len(cols)} != {len(feat_names)}")

        # Expect 24 bins; if not, pad/trim to 24
        T_target = 24
        if T != T_target:
            if T > T_target:
                X = X[-T_target:, :]
            else:
                pad = np.zeros((T_target - T, F), dtype=np.float32)
                X = np.vstack([pad, X])

        for t in range(T_target):
            row = {"stay_id": int(r["stay_id"]), "hour": t}
            for j, name in enumerate(feat_names):
                row[name] = float(X[t, j]) if np.isfinite(X[t, j]) else 0.0
            rows.append(row)

    if feat_names is None:
        raise RuntimeError("No NPZ could be read to infer features.")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    if n_bad:
        print(f"[structured] Skipped {n_bad} NPZ files that could not be read.")
    print(f"[structured] -> {out_path}  ({len(df):,} rows, {len(feat_names)} features)")

def build_images(master: pd.DataFrame, out_path: str):
    # Prefer task-selected image path if present, else fallbacks
    cand_cols = [c for c in [
        "paired_image_path_selected",
        "paired_image_path_48h",
        "paired_image_path_instay",
    ] if c in master.columns]

    rows = []
    if cand_cols:
        img_col = cand_cols[0]
        for _, r in master.iterrows():
            p = r.get(img_col, None)
            if isinstance(p, str) and len(p.strip()):
                rows.append({"stay_id": int(r["stay_id"]), "image_path": p})
    else:
        # No paths were built — write empty table
        pass

    df = pd.DataFrame(rows, columns=["stay_id","image_path"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[images] -> {out_path}  ({len(df):,} rows, column=image_path)")

def build_notes(master: pd.DataFrame, cohort_dir: str, out_path: str):
    # Try to use cohort_with_notes.csv.gz, else notes_master.csv.gz, else empty
    cand = [
        os.path.join(cohort_dir, "cohort_with_notes.csv.gz"),
        os.path.join(cohort_dir, "notes_master.csv.gz"),
    ]
    src = next((p for p in cand if os.path.exists(p)), None)
    rows = []

    if src:
        notes = pd.read_csv(src, compression="gzip")
        # Find chunk columns if present; else a single big text column
        chunk_cols = sorted([c for c in notes.columns if re.match(r"bert_chunk_\d{3}", c)])
        has_text_col = "notes_0_48h" in notes.columns

        # Map by subject_id (notes are per-subject)
        need = "subject_id"
        if need not in notes.columns:
            print(f"[notes] WARNING: {src} missing subject_id; writing empty notes.")
        else:
            notes = notes[[need] + chunk_cols + ([ "notes_0_48h"] if has_text_col else [])]
            subj2texts = {}
            for _, row in notes.iterrows():
                sid = int(row["subject_id"])
                texts = []
                if chunk_cols:
                    for c in chunk_cols:
                        t = row.get(c, None)
                        if isinstance(t, str) and t.strip():
                            texts.append(t)
                elif has_text_col:
                    t = row["notes_0_48h"]
                    if isinstance(t, str) and t.strip():
                        # split into ~512-ish tokens not needed here — one big entry is fine
                        texts.append(t)
                subj2texts[sid] = texts

            for _, r in master.iterrows():
                sid = int(r["subject_id"])
                stay = int(r["stay_id"])
                texts = subj2texts.get(sid, [])
                if not texts:
                    continue
                for t in texts:
                    rows.append({"stay_id": stay, "text": t})

    df = pd.DataFrame(rows, columns=["stay_id","text"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[notes] -> {out_path}  ({len(df):,} rows, column=text)")

def build_labels_and_splits(master: pd.DataFrame, labels_path: str, splits_path: str):
    # labels.parquet: columns mort, pe, ph (pe/ph can be placeholders)
    lab = master[["stay_id","mortality"]].copy()
    lab = lab.rename(columns={"mortality":"mort"})
    lab["pe"] = 0.0
    lab["ph"] = 0.0
    lab = lab.astype({"stay_id":"int64","mort":"float32","pe":"float32","ph":"float32"})
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    lab.to_parquet(labels_path, index=False)
    print(f"[labels] -> {labels_path}  ({len(lab):,} stays)")

    # splits.json
    sp = master[["stay_id","split"]].copy()
    d = {
        "train": [int(x) for x in sp.loc[sp["split"]=="train","stay_id"].tolist()],
        "val":   [int(x) for x in sp.loc[sp["split"]=="val","stay_id"].tolist()],
        "test":  [int(x) for x in sp.loc[sp["split"]=="test","stay_id"].tolist()],
    }
    with open(splits_path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[splits] -> {splits_path}  (train={len(d['train'])}, val={len(d['val'])}, test={len(d['test'])})")

def main():
    ap = argparse.ArgumentParser("Convert datacode.py outputs to Model-ready parquets")
    ap.add_argument("--cohort_dir", required=True, help="Folder with cohort_master.csv.gz and ts_npz/")
    ap.add_argument("--out_root", default="./data", help="Will create <out_root>/MIMIC-IV/*.parquet + splits.json")
    args = ap.parse_args()

    master = read_master(args.cohort_dir)
    mimic_dir = os.path.join(args.out_root, "MIMIC-IV")
    os.makedirs(mimic_dir, exist_ok=True)

    build_structured(master, out_path=os.path.join(mimic_dir, "structured_24h.parquet"))
    build_images(master,    out_path=os.path.join(mimic_dir, "images_24h.parquet"))
    build_notes(master, args.cohort_dir, out_path=os.path.join(mimic_dir, "notes_24h.parquet"))
    build_labels_and_splits(master,
        labels_path=os.path.join(mimic_dir, "labels.parquet"),
        splits_path=os.path.join(mimic_dir, "splits.json"),
    )

if __name__ == "__main__":
    main()
