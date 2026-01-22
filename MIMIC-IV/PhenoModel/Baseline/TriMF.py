from __future__ import annotations
import os as _os
_os.environ.setdefault("HF_HOME", _os.path.expanduser("~/.cache/huggingface"))
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import os
import json
import argparse
import time
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from encoders import BEHRTLabEncoder, BioClinBERTEncoder, ImageEncoder
import torchvision.transforms as T

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from env_config import CFG as _CFG, DEVICE as _DEVICE
    from env_config import load_cfg as _load_cfg
    from env_config import ensure_dir as _ensure_dir
    try:
        from env_config import apply_cli_overrides as _apply_cli_overrides
    except Exception:
        _apply_cli_overrides = None
except Exception:
    _CFG = None
    _DEVICE = _default_device()
    _load_cfg = None
    _ensure_dir = None
    _apply_cli_overrides = None


def set_seed(seed: int) -> None:
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _schema_names(path: Path) -> set:
    path = Path(path)
    if not path.exists():
        return set()

    try:
        import pyarrow.parquet as pq
        if path.is_file():
            return set(pq.read_schema(str(path)).names)
    except Exception:
        pass

    try:
        import pyarrow.dataset as ds
        return set(ds.dataset(str(path), format="parquet").schema.names)
    except Exception:
        pass

    try:
        import fastparquet
        pf = fastparquet.ParquetFile(str(path))
        return set(pf.columns)
    except Exception:
        pass

    try:
        import pandas as pd
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb <= 128:
            df0 = pd.read_parquet(str(path), engine="auto")
            return set(df0.columns)
    except Exception:
        pass

    return set()

def _ensure_parquet_readable(data_root: Path, p: Path, patterns: List[str], tag: str) -> Path:
    def _engine_hint():
        try:
            import pyarrow  
            return "pyarrow=OK"
        except Exception:
            pass
        try:
            import fastparquet  
            return "pyarrow=MISSING, fastparquet=OK"
        except Exception:
            return "pyarrow=MISSING, fastparquet=MISSING (install one)"

    names = _schema_names(p)
    if p.exists() and names:
        return p

    cand = _auto_find_first(data_root, patterns)
    if cand is not None:
        cand_names = _schema_names(cand)
        if cand.exists() and cand_names:
            print(f"[TriMF] {tag}: schema unreadable for {p} -> switched to {cand}")
            return cand

    try:
        st = p.stat()
        size = st.st_size
    except Exception:
        size = None

    raise RuntimeError(
        f"[TriMF][SCHEMA][FAIL] Could not read parquet schema for {tag}: {p} "
        f"(exists={p.exists()} is_file={p.is_file()} size={size}). "
        f"Parquet engine status: {_engine_hint()}. "
        f"Fix: (1) pass the correct --{tag}_parquet path, or (2) install pyarrow/fastparquet."
    )

def _candidate_split_paths(data_root: Path, task: str, split: str) -> List[Path]:
    cands: List[Path] = []
    cands += [
        data_root / task / f"{split}.pt",
        data_root / f"{task}_{split}.pt",
        data_root / f"{split}.pt",
    ]
    cands += [
        data_root / task / f"{split}.pkl",
        data_root / f"{task}_{split}.pkl",
        data_root / f"{split}.pkl",
    ]
    cands += [
        data_root / task / f"{split}.jsonl",
        data_root / f"{task}_{split}.jsonl",
        data_root / f"{split}.jsonl",
    ]
    cands += [
        data_root / task / f"{split}.json",
        data_root / f"{task}_{split}.json",
        data_root / f"{split}.json",
    ]
    seen = set()
    out = []
    for p in cands:
        if str(p) not in seen:
            out.append(p)
            seen.add(str(p))
    return out


def load_split_items(data_root: Union[str, Path], task: str, split: str) -> List[Dict[str, Any]]:
    data_root = Path(data_root)
    for p in _candidate_split_paths(data_root, task, split):
        if p.exists():
            if p.suffix == ".pt":
                obj = torch.load(p, map_location="cpu")
                return _as_list_of_dicts(obj, src=str(p))
            if p.suffix == ".pkl":
                import pickle
                with open(p, "rb") as f:
                    obj = pickle.load(f)
                return _as_list_of_dicts(obj, src=str(p))
            if p.suffix in [".json", ".jsonl"]:
                return _load_json_or_jsonl(p)
    looked = [str(p) for p in _candidate_split_paths(Path(data_root), task, split)]
    raise FileNotFoundError(
        f"Could not find split '{split}' for task='{task}' under data_root={data_root}.\n"
        f"Looked for:\n  - " + "\n  - ".join(looked) + "\n\n"
        f"Fix by placing a split file at one of those locations, or update _candidate_split_paths()."
    )


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return _as_list_of_dicts(obj, src=str(path))


def _as_list_of_dicts(obj: Any, src: str) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], dict):
            return obj
        raise TypeError(f"{src}: expected list[dict] but got list[{type(obj[0])}]")
    if isinstance(obj, dict):
        if "items" in obj and isinstance(obj["items"], list):
            return _as_list_of_dicts(obj["items"], src)
        if all(isinstance(k, (str, int)) for k in obj.keys()) and all(isinstance(v, dict) for v in obj.values()):
            try:
                keys = sorted(obj.keys(), key=lambda x: int(x))
                return [obj[k] for k in keys]
            except Exception:
                return list(obj.values())
        raise TypeError(f"{src}: dict format not recognized (keys: {list(obj.keys())[:10]})")
    raise TypeError(f"{src}: expected list or dict, got {type(obj)}")

def _resolve_path(data_root: Path, p: str) -> Path:
    """Resolve p relative to data_root unless it's already absolute or exists as given."""
    pp = Path(p)
    if pp.is_absolute() and pp.exists():
        return pp
    if pp.exists():
        return pp
    return (Path(data_root) / p)

def _auto_find_first(data_root: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        for p in sorted(data_root.rglob(pat)):
            if p.is_file():
                return p
    return None

def load_splits_json(path: Union[str, Path], task: str) -> Dict[str, List[int]]:
    """Load splits.json and return dict with keys train/val/test -> list[stay_id]."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and task in obj and isinstance(obj[task], dict):
        obj = obj[task]
    if isinstance(obj, dict) and "splits" in obj and isinstance(obj["splits"], dict):
        obj = obj["splits"]

    out: Dict[str, List[int]] = {}
    for split in ["train", "val", "test"]:
        v = obj.get(split, None) if isinstance(obj, dict) else None
        if v is None:
            continue
        if isinstance(v, dict):
            for k in ["stay_id", "ids", "id", "index"]:
                if k in v and isinstance(v[k], list):
                    v = v[k]
                    break
        if not isinstance(v, list):
            raise TypeError(f"{path}: split '{split}' must be a list (or dict wrapping a list), got {type(v)}")
        ids: List[int] = []
        for x in v:
            try:
                ids.append(int(x))
            except Exception:
                pass
        out[split] = ids
    if not out:
        raise ValueError(f"{path}: did not find any of train/val/test splits (task='{task}'). Keys: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}")
    return out


def _chunked(lst: List[Any], n: int) -> List[List[Any]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def read_parquet_filtered(
    path: Union[str, Path],
    ids: List[Any],
    id_col: str = "stay_id",
    columns: Optional[List[str]] = None,
    chunk_size: int = 50000,
) -> "pd.DataFrame":
    import pandas as pd
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if len(ids) == 0:
        return pd.DataFrame(columns=(columns or []))

    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.types as pat

        dataset = ds.dataset(str(path), format="parquet")
        schema_names = set(dataset.schema.names)
        if id_col not in schema_names:
            raise KeyError(f"{path.name}: id_col='{id_col}' not in parquet schema. Available: {sorted(list(schema_names))[:40]}")
        col_type = dataset.schema.field(id_col).type

        def _coerce_ids(chunk):
            if pat.is_string(col_type) or pat.is_large_string(col_type):
                return pa.array([str(x) for x in chunk], type=col_type)
            if pat.is_integer(col_type):
                return pa.array([int(x) for x in chunk], type=col_type)
            if pat.is_floating(col_type):
                return pa.array([float(x) for x in chunk], type=col_type)
            return pa.array(chunk)

        tables = []
        for chunk in _chunked(list(ids), chunk_size):
            ids_arr = _coerce_ids(chunk)
            filt = ds.field(id_col).isin(ids_arr)
            tables.append(dataset.to_table(filter=filt, columns=columns))

        if not tables:
            return pd.DataFrame(columns=(columns or []))
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        return table.to_pandas()

    except Exception:
        df = pd.read_parquet(path, columns=columns)
        if id_col not in df.columns:
            raise KeyError(f"{path.name}: id_col='{id_col}' not in columns. Available: {list(df.columns)[:40]}")

        if df[id_col].dtype == object:
            ids2 = set(str(x) for x in ids)
            return df[df[id_col].astype(str).isin(ids2)]
        else:
            return df[df[id_col].isin(ids)]

def _infer_id_col(df_cols: List[str]) -> str:
    for c in ["stay_id", "hadm_id", "subject_id", "id"]:
        if c in df_cols:
            return c
    return df_cols[0] if df_cols else "stay_id"



def _pick_filter_key(path: Path, stay_id_col: str, meta_df: "pd.DataFrame") -> Optional[str]:
    names = _schema_names(path)
    meta_cols = set(meta_df.columns)

    candidates = [stay_id_col, "stay_id", "icustay_id", "hadm_id", "subject_id", "id"]
    for k in candidates:
        if k in names and k in meta_cols:
            return k

    if stay_id_col in names:
        return stay_id_col

    return None

def _read_modality_df(path: Path, stay_ids: List[int], stay_id_col: str, meta_df: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd
    if not path.exists():
        return pd.DataFrame()

    key = _pick_filter_key(path, stay_id_col=stay_id_col, meta_df=meta_df)
    if key is None:
        print(f"[TriMF][JOIN][WARN] {path.name}: no usable key. parquet_cols={sorted(list(_schema_names(path)))[:30]} meta_cols={list(meta_df.columns)}")
        return pd.DataFrame()

    if key == stay_id_col:
        filt_ids = stay_ids
    else:
        filt_ids = meta_df[key].dropna().tolist()
        filt_ids = [x.item() if hasattr(x, "item") else x for x in filt_ids]
        if len(filt_ids) == 0:
            return pd.DataFrame()

    df = read_parquet_filtered(path, filt_ids, id_col=key)
    if df.empty:
        return df
    if key != stay_id_col:
        df = df.merge(
            meta_df[[stay_id_col, key]].dropna().drop_duplicates(),
            on=key,
            how="inner",
        )
    return df


def _pick_first_key(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _extract_labs_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    if isinstance(row, dict) and "x" in row:
        return {"x": row["x"]}

    if "labs" in row and isinstance(row["labs"], dict):
        return row["labs"]

    in_ids = _pick_first_key(row, ["input_ids", "behrt_input_ids", "lab_input_ids", "labs_input_ids", "input_ids_l"])
    attn  = _pick_first_key(row, ["attention_mask", "behrt_attention_mask", "lab_attention_mask", "labs_attention_mask", "attention_mask_l"])
    ttype = _pick_first_key(row, ["token_type_ids", "behrt_token_type_ids", "lab_token_type_ids", "token_type_ids_l"])

    if in_ids is None:
        return None

    out = {"input_ids": in_ids}
    if attn is not None:
        out["attention_mask"] = attn
    if ttype is not None:
        out["token_type_ids"] = ttype
    return out

def _extract_notes_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    if "notes" in row and isinstance(row["notes"], dict):
        return row["notes"]

    in_ids = _pick_first_key(row, ["input_ids", "note_input_ids", "notes_input_ids", "input_ids_n"])
    attn  = _pick_first_key(row, ["attention_mask", "note_attention_mask", "notes_attention_mask", "attention_mask_n"])
    cmask = _pick_first_key(row, ["chunk_mask", "note_chunk_mask", "notes_chunk_mask", "chunk_mask_n"])

    if in_ids is not None:
        out = {"input_ids": in_ids}
        if attn is not None:
            out["attention_mask"] = attn
        if cmask is not None:
            out["chunk_mask"] = cmask
        return out

    txt = _pick_first_key(row, ["text", "note_text", "notes_text", "TEXTCHUNKS", "text_chunks", "chunks_text"])
    if txt is not None:
        return {"text": txt}

    return None


def _extract_image_from_row(row: Dict[str, Any]) -> Optional[Any]:
    if row is None:
        return None
    if "image" in row:
        return row["image"]

    p = _pick_first_key(row, [
        "image_path", "img_path", "cxr_path", "path", "filepath", "file_path",
        "png_path", "jpg_path", "jpeg_path", "dicom_path"
    ])
    if p is not None:
        if isinstance(p, (list, tuple)) and len(p) > 0:
            return p[0]
        return p

    b = _pick_first_key(row, ["png", "jpg", "jpeg", "bytes", "image_bytes"])
    if b is not None:
        return b
    return None


def _suffix_int(name: str) -> Optional[int]:
    try:
        return int(name.split("_")[-1])
    except Exception:
        return None

def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, (np.floating, float)) and np.isnan(v):
            return default
        return int(v)
    except Exception:
        return default

def load_items_from_parquet_stack(args, split: str, num_labels: int) -> List[Dict[str, Any]]:
    """Build list[dict] samples using splits.json + {structured,notes,images,labels} parquets."""
    import pandas as pd

    data_root = Path(args.data_root)
    struct_path = _resolve_path(data_root, args.structured_parquet)
    if not struct_path.exists():
        cand = _auto_find_first(data_root, ["*xehr*.parquet", "*structured*.parquet", "*labs*.parquet"])
        if cand is not None:
            print(f"[TriMF] Auto-picked structured_parquet: {cand}")
            struct_path = cand
        else:
            print(f"[TriMF][WARN] structured_parquet not found: {struct_path} (will treat labs as missing)")

    notes_path = _resolve_path(data_root, args.notes_parquet)
    if not notes_path.exists():
        cand = _auto_find_first(data_root, ["*notes*.parquet", "*TEXTCHUNKS*.parquet", "*note*.parquet"])
        if cand is not None:
            print(f"[TriMF] Auto-picked notes_parquet: {cand}")
            notes_path = cand
        else:
            print(f"[TriMF][WARN] notes_parquet not found: {notes_path} (will treat notes as missing)")

    img_path = _resolve_path(data_root, args.images_parquet)
    if not img_path.exists():
        cand = _auto_find_first(data_root, ["*images*.parquet", "*image*.parquet"])
        if cand is not None:
            print(f"[TriMF] Auto-picked images_parquet: {cand}")
            img_path = cand
        else:
            print(f"[TriMF][WARN] images_parquet not found: {img_path} (will treat images as missing)")

    if struct_path.exists():
        struct_path = _ensure_parquet_readable(
            data_root, struct_path,
            patterns=["*xehr*.parquet", "*structured*.parquet", "*labs*.parquet"],
            tag="structured",
        )

    if notes_path.exists():
        notes_path = _ensure_parquet_readable(
            data_root, notes_path,
            patterns=["*notes*.parquet", "*TEXTCHUNKS*.parquet", "*note*.parquet"],
            tag="notes",
        )

    if img_path.exists():
        img_path = _ensure_parquet_readable(
            data_root, img_path,
            patterns=["*images*.parquet", "*image*.parquet"],
            tag="images",
        )


    splits_path = _resolve_path(data_root, args.splits_json)
    if not splits_path.exists():
        cand = _auto_find_first(data_root, ["*splits*.json", "*split*.json"])
        if cand is None:
            raise FileNotFoundError(f"Could not find splits json (tried {splits_path})")
        print(f"[TriMF] Auto-picked splits_json: {cand}")
        splits_path = cand
    splits = load_splits_json(splits_path, args.task)
    if split not in splits:
        raise KeyError(f"splits.json has no '{split}' split. Available: {list(splits.keys())}")
    stay_ids = splits[split]
    if len(stay_ids) == 0:
        return []

    labels_path = _resolve_path(data_root, args.labels_parquet)
    if not labels_path.exists():
        cand = _auto_find_first(data_root, [f"*labels*{args.task}*.parquet", "*labels*pheno*.parquet", "*labels*.parquet"])
        if cand is None:
            raise FileNotFoundError(f"Could not find labels parquet (tried {labels_path})")
        print(f"[TriMF] Auto-picked labels_parquet: {cand}")
        labels_path = cand

    lab_df = read_parquet_filtered(labels_path, stay_ids, id_col=args.id_col)
    if lab_df.empty:
        raise RuntimeError(f"No rows loaded from labels parquet for split '{split}'. path={labels_path}")
    id_col = _infer_id_col(list(lab_df.columns))

    meta_keep = [id_col] + [c for c in ["icustay_id", "hadm_id", "subject_id"] if c in lab_df.columns]
    meta_df = lab_df[meta_keep].drop_duplicates().copy()

    meta_cols = set([id_col, "subject_id", "hadm_id", "icustay_id"])
    label_cols = [c for c in lab_df.columns if c not in meta_cols]
    if len(label_cols) != num_labels:
        pref = [c for c in label_cols if c.lower().startswith(("pheno", "phe", "phenotype", "label", "y"))]
        if len(pref) == num_labels:
            label_cols = pref
        else:
            label_cols = label_cols[:num_labels]

    def _augment_meta(meta_df: "pd.DataFrame", p: Path, stay_ids: List[int], stay_id_col: str) -> "pd.DataFrame":
        if not p.exists():
            return meta_df
        names = _schema_names(p)
        if not names or stay_id_col not in names:
            return meta_df

        add_cols = [c for c in ["hadm_id", "subject_id", "icustay_id"] if c in names and c not in meta_df.columns]
        if not add_cols:
            return meta_df

        cols = [stay_id_col] + add_cols
        m = read_parquet_filtered(p, stay_ids, id_col=stay_id_col, columns=cols)
        if m.empty:
            return meta_df

        m = m[cols].dropna().drop_duplicates()
        return meta_df.merge(m, on=stay_id_col, how="left")

    meta_df = _augment_meta(meta_df, struct_path, stay_ids, stay_id_col=id_col)
    meta_df = _augment_meta(meta_df, notes_path,  stay_ids, stay_id_col=id_col)
    meta_df = _augment_meta(meta_df, img_path,    stay_ids, stay_id_col=id_col)
    print(f"[TriMF][LABELS] inferred id_col={id_col} | label_cols={len(label_cols)}")
    print(f"[TriMF][META] meta_df cols: {list(meta_df.columns)}")
    print(f"[TriMF][SCHEMA] structured: {sorted(list(_schema_names(struct_path)))[:30]}")
    print(f"[TriMF][SCHEMA] notes:      {sorted(list(_schema_names(notes_path)))[:30]}")
    print(f"[TriMF][SCHEMA] images:     {sorted(list(_schema_names(img_path)))[:30]}")


    meta_cols = set([id_col, "subject_id", "hadm_id", "icustay_id"])
    label_cols = [c for c in lab_df.columns if c not in meta_cols]
    if len(label_cols) != num_labels:
        pref = [c for c in label_cols if c.lower().startswith(("pheno", "phe", "phenotype", "label", "y"))]
        if len(pref) == num_labels:
            label_cols = pref
        else:
            label_cols = label_cols[:num_labels]
    labels_map = {}
    for _, r in lab_df.iterrows():
        sid = int(r[id_col])
        labels_map[sid] = r[label_cols].to_numpy(dtype=np.float32)

    struct_map = {}
    if struct_path.exists():
        s_df = _read_modality_df(struct_path, stay_ids, stay_id_col=id_col, meta_df=meta_df)
        if not s_df.empty and id_col in s_df.columns:
            drop_cols = {id_col, "subject_id", "hadm_id", "icustay_id"}
            feat_cols = [c for c in s_df.columns if c not in drop_cols]

            print(f"[TriMF][STRUCT] rows={len(s_df)} | uniq={s_df[id_col].nunique()} | feat_dim={len(feat_cols)}")

            for sid, g in s_df.groupby(id_col, sort=False):
                r = g.iloc[0]
                x = r[feat_cols].to_numpy(dtype=np.float32, copy=True)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                struct_map[int(sid)] = {"x": x}

    notes_map = {}
    if notes_path.exists():
        n_df = _read_modality_df(notes_path, stay_ids, stay_id_col=id_col, meta_df=meta_df)

        if not n_df.empty and id_col in n_df.columns:
            wide_ids = sorted([c for c in n_df.columns if c.startswith("input_ids_")], key=lambda x: _suffix_int(x) or 10**9)
            wide_attn = sorted([c for c in n_df.columns if c.startswith("attn_mask_") or c.startswith("attention_mask_")], key=lambda x: _suffix_int(x) or 10**9)

            if wide_ids:
                print(f"[TriMF][NOTES] wide tokens: ids_cols={len(wide_ids)} attn_cols={len(wide_attn)} rows={len(n_df)} uniq={n_df[id_col].nunique()}")
                for sid, g in n_df.groupby(id_col, sort=False):
                    chunks, attns = [], []
                    for _, r in g.iterrows():
                        ids = [_safe_int(v, 0) for v in r[wide_ids].tolist()]
                        chunks.append(ids)
                        if wide_attn:
                            am = [_safe_int(v, 0) for v in r[wide_attn].tolist()]
                            attns.append(am)

                    notes_map[int(sid)] = {
                        "input_ids": chunks,
                        "attention_mask": attns if wide_attn else None,
                        "chunk_mask": [1] * len(chunks),
                    }

            elif "input_ids" in n_df.columns:
                if n_df.duplicated(id_col).any():
                    for sid, g in n_df.groupby(id_col, sort=False):
                        sid = int(sid)
                        chunks = [x for x in g["input_ids"].tolist() if x is not None]
                        attns  = [x for x in (g["attention_mask"].tolist() if "attention_mask" in g.columns else []) if x is not None]
                        notes_map[sid] = {
                            "input_ids": chunks,
                            "attention_mask": attns if len(attns) == len(chunks) else None,
                            "chunk_mask": [1] * len(chunks),
                        }
                else:
                    for _, r in n_df.iterrows():
                        notes_map[int(r[id_col])] = r.to_dict()

            else:
                text_col = None
                for cand in ["TEXTCHUNKS", "text_chunks", "text", "note_text", "notes_text", "chunks_text"]:
                    if cand in n_df.columns:
                        text_col = cand
                        break

                if text_col is not None:
                    for sid, g in n_df.groupby(id_col, sort=False):
                        sid = int(sid)
                        vals = [v for v in g[text_col].tolist() if v is not None]
                        chunks: List[str] = []
                        for v in vals:
                            if isinstance(v, list):
                                chunks.extend([str(s) for s in v if s])
                            else:
                                chunks.append(str(v))
                        notes_map[sid] = {"text": "\n".join(chunks) if chunks else ""}
                else:
                    for sid, g in n_df.groupby(id_col, sort=False):
                        notes_map[int(sid)] = g.iloc[0].to_dict()

    img_map = {}
    if img_path.exists():
        i_df = _read_modality_df(img_path, stay_ids, stay_id_col=id_col, meta_df=meta_df)
        if not i_df.empty and id_col in i_df.columns:
            for sid, g in i_df.groupby(id_col, sort=False):
                img_map[int(sid)] = g.iloc[0].to_dict()


    items: List[Dict[str, Any]] = []
    miss_y = 0
    for sid in stay_ids:
        y = labels_map.get(int(sid), None)
        if y is None:
            miss_y += 1
            continue

        struct_row = struct_map.get(int(sid), None)
        notes_row = notes_map.get(int(sid), None)
        img_row   = img_map.get(int(sid), None)

        item = {
            "stay_id": int(sid),
            "y": y,
            "labs": _extract_labs_from_row(struct_row) if struct_row is not None else None,
            "notes": _extract_notes_from_row(notes_row) if notes_row is not None else None,
            "image": _extract_image_from_row(img_row) if img_row is not None else None,
        }
        items.append(item)

    if miss_y > 0:
        print(f"[WARN] split={split}: {miss_y} stay_ids had no labels in {labels_path.name} and were skipped.")
    return items

def load_split_items_auto(args, split: str, num_labels: int) -> List[Dict[str, Any]]:
    """Try cached split files first; fallback to splits.json + parquet stack."""
    try:
        return load_split_items(args.data_root, args.task, split)
    except FileNotFoundError:
        return load_items_from_parquet_stack(args, split=split, num_labels=num_labels)

def _canonize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for k in ["stay_id", "id", "sample_id", "hadm_id"]:
        if k in item:
            out["id"] = item[k]
            break

    for k in ["y", "label", "labels", "target", "targets", "pheno", "phenotypes"]:
        if k in item:
            out["y"] = item[k]
            break
    if "y" not in out:
        raise KeyError(f"Item missing labels. Keys present: {list(item.keys())}")

    if "labs" in item:
        out["labs"] = item["labs"]
    elif "lab" in item:
        out["labs"] = item["lab"]
    else:
        lab_keys = ["lab_input_ids", "lab_attention_mask", "lab_token_type_ids",
                    "behrt_input_ids", "behrt_attention_mask", "behrt_token_type_ids",
                    "input_ids_l", "attention_mask_l", "token_type_ids_l"]
        if any(k in item for k in lab_keys):
            labs = {}
            for src_k, dst_k in [
                ("lab_input_ids", "input_ids"),
                ("behrt_input_ids", "input_ids"),
                ("input_ids_l", "input_ids"),
                ("lab_attention_mask", "attention_mask"),
                ("behrt_attention_mask", "attention_mask"),
                ("attention_mask_l", "attention_mask"),
                ("lab_token_type_ids", "token_type_ids"),
                ("behrt_token_type_ids", "token_type_ids"),
                ("token_type_ids_l", "token_type_ids"),
            ]:
                if src_k in item:
                    labs[dst_k] = item[src_k]
            out["labs"] = labs
        else:
            out["labs"] = None  

    if "notes" in item:
        out["notes"] = item["notes"]
    elif "note" in item:
        out["notes"] = item["note"]
    else:
        note_keys = ["note_input_ids", "note_attention_mask", "note_chunk_mask",
                     "notes_input_ids", "notes_attention_mask", "notes_chunk_mask",
                     "input_ids_n", "attention_mask_n", "chunk_mask_n",
                     "note_text", "text"]
        if any(k in item for k in note_keys):
            notes = {}
            for src_k, dst_k in [
                ("notes_input_ids", "input_ids"),
                ("note_input_ids", "input_ids"),
                ("input_ids_n", "input_ids"),
                ("notes_attention_mask", "attention_mask"),
                ("note_attention_mask", "attention_mask"),
                ("attention_mask_n", "attention_mask"),
                ("notes_chunk_mask", "chunk_mask"),
                ("note_chunk_mask", "chunk_mask"),
                ("chunk_mask_n", "chunk_mask"),
                ("note_text", "text"),
                ("text", "text"),
            ]:
                if src_k in item:
                    notes[dst_k] = item[src_k]
            out["notes"] = notes
        else:
            out["notes"] = None

    for k in ["image", "img", "image_path", "img_path", "path_i"]:
        if k in item:
            out["image"] = item[k]
            break
    if "image" not in out:
        out["image"] = None

    return out



def _max_in_nested_int(x: Any) -> Optional[int]:
    """Return max int in possibly nested (list/tuple/ndarray/tensor) structure. None if empty/invalid."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return None
        try:
            return int(x.max().item())
        except Exception:
            return None
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        if np.issubdtype(x.dtype, np.number):
            try:
                return int(np.nanmax(x))
            except Exception:
                return None
        return _max_in_nested_int(x.tolist())
    if isinstance(x, (list, tuple)):
        m = None
        for e in x:
            v = _max_in_nested_int(e)
            if v is None:
                continue
            m = v if m is None else max(m, v)
        return m
    try:
        return int(x)
    except Exception:
        return None


def infer_n_feats_from_items(items: List[Dict[str, Any]], fallback: Optional[int] = None) -> int:
    mx: Optional[int] = None
    for it in items:
        try:
            c = _canonize_item(it)
            labs = c.get("labs", None)
            if labs is None:
                continue
            ids = labs.get("input_ids", None) if isinstance(labs, dict) else labs
            v = _max_in_nested_int(ids)
            if v is None:
                continue
            mx = v if mx is None else max(mx, v)
        except Exception:
            continue
    if mx is None:
        return int(fallback) if (fallback is not None and int(fallback) > 0) else 1
    # If ids are 0-indexed, vocab size is max+1
    nf = int(mx) + 1
    if nf <= 0:
        nf = int(fallback) if (fallback is not None and int(fallback) > 0) else 1
    return nf


class MultimodalCacheDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], num_labels: int):
        self.items = items
        self.num_labels = num_labels

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = _canonize_item(self.items[idx])
        return item


def _to_1d_long(x: Any) -> torch.Tensor:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.long().view(-1)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).long().view(-1)
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.long).view(-1)
    return torch.tensor(np.asarray(x), dtype=torch.long).view(-1)


def _to_2d_long(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        t = x.long()
        return t.view(1, -1) if t.ndim == 1 else t
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x).long()
        return t.view(1, -1) if t.ndim == 1 else t
    if isinstance(x, list):
        if len(x) == 0:
            return torch.zeros((1, 1), dtype=torch.long)

        try:
            t = torch.tensor(x, dtype=torch.long)
            return t.view(1, -1) if t.ndim == 1 else t
        except Exception:
            rows: List[torch.Tensor] = []
            maxlen = 0
            for r in x:
                rt = _to_1d_long(r)
                if rt is None or rt.numel() == 0:
                    rt = torch.zeros(1, dtype=torch.long)
                rows.append(rt)
                maxlen = max(maxlen, rt.numel())

            out = torch.zeros((len(rows), maxlen), dtype=torch.long)
            for i, rt in enumerate(rows):
                out[i, : rt.numel()] = rt
            return out

    arr = np.asarray(x)
    t = torch.tensor(arr, dtype=torch.long)
    return t.view(1, -1) if t.ndim == 1 else t

def _to_float_labels(y: Any, num_labels: int) -> torch.Tensor:
    if isinstance(y, torch.Tensor):
        t = y.float()
    elif isinstance(y, np.ndarray):
        t = torch.from_numpy(y).float()
    else:
        t = torch.tensor(y, dtype=torch.float32)
    if t.ndim == 0:
        t = t.view(1)
    if t.numel() == 1 and num_labels > 1:
        raise ValueError(f"Got scalar label but num_labels={num_labels}. Provide a {num_labels}-dim multi-hot vector.")
    if t.numel() != num_labels:
        raise ValueError(f"Label size mismatch: got {t.numel()}, expected {num_labels}.")
    return t.view(num_labels)


def _pad_1d(seqs: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    max_len = max([s.numel() for s in seqs]) if seqs else 0
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : s.numel()] = s
    return out


def _pad_3d(seqs: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    max_c = max([s.shape[0] for s in seqs]) if seqs else 0
    max_l = max([s.shape[1] for s in seqs]) if seqs else 0
    out = torch.full((len(seqs), max_c, max_l), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        c, l = s.shape
        out[i, :c, :l] = s
    return out


def build_image_transform(image_size: int, img_norm: str) -> T.Compose:
    tfms: List[Any] = [
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ]
    if img_norm.lower() == "imagenet":
        tfms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]))
    elif img_norm.lower() == "none":
        pass
    else:
        raise ValueError(f"--img_norm must be 'imagenet' or 'none', got {img_norm}")
    return T.Compose(tfms)


def _load_image_any(x: Any, roots: Optional[List[Union[str, Path]]] = None) -> Optional[Image.Image]:
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, str):
        p = Path(x)
        if p.exists():
            return Image.open(p).convert("RGB")
        if roots:
            for r in roots:
                if r is None:
                    continue
                rr = Path(r)
                pp = rr / x
                if pp.exists():
                    return Image.open(pp).convert("RGB")
        return None
    return None



def collate_multimodal(
    batch: List[Dict[str, Any]],
    num_labels: int,
    image_tfm: Optional[T.Compose] = None,
    image_roots: Optional[List[Union[str, Path]]] = None,
    skip_missing_images: bool = False,
) -> Dict[str, Any]:

    ids = [b.get("id", None) for b in batch]
    y = torch.stack([_to_float_labels(b["y"], num_labels) for b in batch], dim=0)
    labs_list = [b.get("labs", None) for b in batch]
    have_labs = [x is not None for x in labs_list]

    labs_batch = None
    if any(have_labs):
        if any(isinstance(x, dict) and ("x" in x) for x in labs_list if x is not None):
            vecs: List[torch.Tensor] = []
            max_dim = 0
            for x in labs_list:
                if x is None or not (isinstance(x, dict) and "x" in x) or x["x"] is None:
                    t = torch.zeros(1, dtype=torch.float32)
                else:
                    arr = x["x"]
                    if isinstance(arr, torch.Tensor):
                        t = arr.float().view(-1)
                    elif isinstance(arr, np.ndarray):
                        t = torch.from_numpy(arr).float().view(-1)
                    else:
                        t = torch.tensor(arr, dtype=torch.float32).view(-1)
                max_dim = max(max_dim, t.numel())
                vecs.append(t)

            X = torch.zeros((len(vecs), max_dim), dtype=torch.float32)
            for i, t in enumerate(vecs):
                X[i, : t.numel()] = t
            labs_batch = {"x": X}

        else:
            lab_ids: List[torch.Tensor] = []
            lab_attn: List[torch.Tensor] = []
            lab_ttype: List[Optional[torch.Tensor]] = []

            for x in labs_list:
                if x is None:
                    ids_t = torch.zeros(1, dtype=torch.long)
                    attn_t = torch.zeros(1, dtype=torch.long)
                    ttype_t = None
                elif isinstance(x, dict):
                    ids_t = _to_1d_long(x.get("input_ids", x.get("ids", x.get("tokens"))))
                    if ids_t is None or ids_t.numel() == 0:
                        ids_t = torch.zeros(1, dtype=torch.long)

                    attn_t = _to_1d_long(x.get("attention_mask", x.get("mask")))
                    if attn_t is None or attn_t.numel() == 0:
                        attn_t = torch.ones_like(ids_t)

                    ttype_t = _to_1d_long(x.get("token_type_ids", None))
                    if ttype_t is not None and ttype_t.numel() == 0:
                        ttype_t = None
                else:
                    ids_t = _to_1d_long(x)
                    if ids_t is None or ids_t.numel() == 0:
                        ids_t = torch.zeros(1, dtype=torch.long)
                    attn_t = torch.ones_like(ids_t)
                    ttype_t = None

                lab_ids.append(ids_t)
                lab_attn.append(attn_t)
                lab_ttype.append(ttype_t)

            lab_input_ids = _pad_1d(lab_ids, pad_value=0)
            lab_attention_mask = _pad_1d(lab_attn, pad_value=0)

            labs_batch = {
                "input_ids": lab_input_ids,
                "attention_mask": lab_attention_mask,
            }
            if any(t is not None for t in lab_ttype):
                tt = [(t if t is not None else torch.zeros(1, dtype=torch.long)) for t in lab_ttype]
                labs_batch["token_type_ids"] = _pad_1d(tt, pad_value=0)

    notes_list = [b.get("notes", None) for b in batch]
    have_notes = [x is not None for x in notes_list]
    if any(have_notes):
        note_ids_3d: List[torch.Tensor] = []
        note_attn_3d: List[torch.Tensor] = []
        chunk_masks: List[torch.Tensor] = []
        raw_texts: List[Optional[Any]] = []

        for x in notes_list:
            if x is None:
                ids = torch.zeros((1, 1), dtype=torch.long)
                attn = torch.zeros((1, 1), dtype=torch.long)
                cm_t = torch.zeros((1,), dtype=torch.long)
                raw_texts.append(None)

            elif isinstance(x, dict) and "text" in x and (("input_ids" not in x) or (x["input_ids"] is None)):
                raw_texts.append(x["text"])
                ids = torch.zeros((1, 1), dtype=torch.long)
                attn = torch.zeros((1, 1), dtype=torch.long)
                cm_t = torch.zeros((1,), dtype=torch.long)

            else:
                raw_texts.append(None)

                if isinstance(x, dict):
                    ids = _to_2d_long(x.get("input_ids", x.get("ids")))
                    if ids is None or ids.numel() == 0:
                        ids = torch.zeros((1, 1), dtype=torch.long)

                    attn = _to_2d_long(x.get("attention_mask", x.get("mask")))
                    if attn is None or attn.numel() == 0:
                        attn = torch.ones_like(ids)

                    cm = x.get("chunk_mask", None)
                    if cm is None:
                        cm_t = torch.ones((ids.shape[0],), dtype=torch.long)
                    else:
                        cm_t = _to_1d_long(cm)
                        if cm_t is None or cm_t.numel() == 0:
                            cm_t = torch.ones((ids.shape[0],), dtype=torch.long)

                else:
                    ids = _to_2d_long(x)
                    if ids is None or ids.numel() == 0:
                        ids = torch.zeros((1, 1), dtype=torch.long)
                    attn = torch.ones_like(ids)
                    cm_t = torch.ones((ids.shape[0],), dtype=torch.long)

            if ids.ndim == 1:
                ids = ids.view(1, -1)
            if attn.ndim == 1:
                attn = attn.view(1, -1)

            if attn.shape != ids.shape:
                attn = torch.ones_like(ids)

            if cm_t.ndim == 0:
                cm_t = cm_t.view(1)
            if cm_t.numel() != ids.shape[0]:
                cm_t = torch.ones((ids.shape[0],), dtype=torch.long)

            note_ids_3d.append(ids)
            note_attn_3d.append(attn)
            chunk_masks.append(cm_t)

        notes_input_ids = _pad_3d(note_ids_3d, pad_value=0)
        notes_attention_mask = _pad_3d(note_attn_3d, pad_value=0)

        max_c = notes_input_ids.shape[1]
        cm_pad = torch.zeros((len(batch), max_c), dtype=torch.long)
        for i, cm in enumerate(chunk_masks):
            cm_pad[i, : cm.numel()] = cm

        notes_batch = {
            "input_ids": notes_input_ids,
            "attention_mask": notes_attention_mask,
            "chunk_mask": cm_pad,
            "raw_text": raw_texts,
        }
    else:
        notes_batch = None

    imgs_list = [b.get("image", None) for b in batch]
    have_imgs = [x is not None for x in imgs_list]
    if any(have_imgs):
        if image_tfm is None:
            image_tfm = build_image_transform(224, "imagenet")

        dummy = Image.new("RGB", (256, 256))
        ph = image_tfm(dummy)
        if not (isinstance(ph, torch.Tensor) and ph.ndim == 3):
            raise ValueError("image_tfm must produce a [3,H,W] tensor.")
        ph_shape = tuple(ph.shape)  # (3,H,W)

        img_tensors: List[torch.Tensor] = []
        for x in imgs_list:
            if x is None:
                img_tensors.append(torch.zeros(ph_shape, dtype=torch.float32))
                continue
            if isinstance(x, torch.Tensor):
                t = x.float()
                if t.ndim == 3:
                    img_tensors.append(t)
                elif t.ndim == 4:
                    img_tensors.append(t[0])
                else:
                    raise ValueError(f"Unexpected image tensor shape: {t.shape}")
                continue

            pil = _load_image_any(x, roots=image_roots)
            if pil is None:
                if skip_missing_images:
                    img_tensors.append(torch.zeros(ph_shape, dtype=torch.float32))
                    continue
                raise FileNotFoundError(
                    f"Image path not found: {x}. "
                    f"Fix by setting --image_root to a directory that contains this relative path."
                )
            img_tensors.append(image_tfm(pil))


        images = torch.stack(img_tensors, dim=0)
    else:
        images = None

    return {"id": ids, "labs": labs_batch, "notes": notes_batch, "images": images, "y": y}

def _tensor_from_encoder_out(out: Any) -> torch.Tensor:
    """Best-effort extraction of the primary tensor from various encoder return types."""
    if out is None:
        raise ValueError("Encoder returned None.")
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for k in [
            "chunk_emb", "chunk_embedding", "chunk_reps", "chunk_repr",
            "last_hidden_state", "sequence_output", "hidden_states",
            "pooled", "pooler_output", "embedding", "emb", "h", "rep", "repr", "z",
        ]:
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
        tv = [v for v in out.values() if isinstance(v, torch.Tensor)]
        if not tv:
            raise ValueError(f"Encoder dict output has no tensors. Keys: {list(out.keys())}")
        return tv[0]
    if isinstance(out, (tuple, list)):
        for v in out:
            if isinstance(v, torch.Tensor):
                return v
        raise ValueError("Encoder tuple/list output had no tensors.")
    raise TypeError(f"Unsupported encoder output type: {type(out)}")


def _pool_tokens(t: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pool token sequences [B,T,D] into [B,D] (masked mean if mask provided)."""
    if t.ndim != 3:
        raise ValueError(f"_pool_tokens expects [B,T,D], got {t.shape}")
    if attention_mask is not None and attention_mask.ndim == 2:
        am = attention_mask.float()
        denom = am.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (t * am.unsqueeze(-1)).sum(dim=1) / denom
    return t[:, 0, :]


def _pick_emb_from_encoder_out(out: Any, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    t = _tensor_from_encoder_out(out)

    if t.ndim == 3:
        return _pool_tokens(t, attention_mask=attention_mask)
    if t.ndim == 2:
        return t
    if t.ndim == 4:
        return t.flatten(1)
    raise ValueError(f"Unexpected tensor shape from encoder: {t.shape}")


class PairFusion(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(4 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = self.norm(a)
        b = self.norm(b)
        feats = torch.cat([a, b, a * b, torch.abs(a - b)], dim=-1)
        h = self.mlp(feats)
        h = h + 0.5 * (a + b)
        return self.out_norm(h)


class TriFusion(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(4 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
            nn.Dropout(dropout),
            nn.LayerNorm(d),
        )

    def forward(self, ln: torch.Tensor, li: torch.Tensor, ni: torch.Tensor) -> torch.Tensor:
        s_ln = self.gate(ln)
        s_li = self.gate(li)
        s_ni = self.gate(ni)
        scores = torch.cat([s_ln, s_li, s_ni], dim=1)  # [B,3]
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B,3,1]
        stack = torch.stack([ln, li, ni], dim=1)  # [B,3,D]
        wsum = (w * stack).sum(dim=1)  # [B,D]
        feats = torch.cat([wsum, ln, li, ni], dim=-1)
        return self.mlp(feats)


def _instantiate_encoder(enc_cls, cfg: Any, d_model: int, extra: Optional[Dict[str, Any]] = None):
    extra = extra or {}
    import inspect

    enc_name = enc_cls.__name__.lower()
    is_image = ("image" in enc_name) or ("img" in enc_name) or ("vision" in enc_name)
    is_text  = ("bert" in enc_name) or ("text" in enc_name) or ("note" in enc_name) or ("clin" in enc_name)

    cfg_names = {"cfg", "config", "args"}
    d_names = {"d", "dim", "d_model", "hidden", "hidden_size", "out_dim", "embed_dim", "proj_dim"}
    model_names = {"model_name", "pretrained", "pretrained_model_name", "bert_name", "hf_model", "text_model_name", "image_model_name"}
    nfeat_names = {"n_feats", "num_feats", "vocab_size", "structured_n_feats"}

    def _pick_text_model() -> Optional[str]:
        return (
            getattr(cfg, "text_model_name", None)
            or getattr(cfg, "bert_model_name", None)
            or getattr(cfg, "language_model_name", None)
            or getattr(cfg, "model_name", None)
        )

    def _pick_image_model() -> Optional[str]:
        return (
            getattr(cfg, "image_model_name", None)
            or getattr(cfg, "img_model_name", None)
            or getattr(cfg, "vision_model_name", None)
            or getattr(cfg, "model_name", None)
        )

    def _get_value(name: str) -> Any:
        if name in extra and extra[name] is not None:
            return extra[name]
        if name in cfg_names:
            return cfg
        if name in d_names:
            return d_model
        if name in nfeat_names:
            return (
                extra.get("n_feats", None)
                or getattr(cfg, "structured_n_feats", None)
                or getattr(cfg, "n_feats", None)
            )
        if name in model_names:
            return _pick_image_model() if is_image else _pick_text_model()
        if hasattr(cfg, name):
            return getattr(cfg, name)
        return None

    sig = inspect.signature(enc_cls.__init__)
    params = list(sig.parameters.values())[1:]  # skip self
    req = [p for p in params if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) and p.default is inspect._empty]

    kwargs: Dict[str, Any] = {}
    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        v = _get_value(p.name)
        if v is None:
            continue
        kwargs[p.name] = v

    missing = [p.name for p in req if p.name not in kwargs]
    last_type_error: Optional[TypeError] = None

    if not missing:
        try:
            inst = enc_cls(**kwargs)
            print(f"[TriMF] Initialized {enc_cls.__name__} with signature kwargs: {sorted(kwargs.keys())}")
            return inst
        except TypeError as e:
            last_type_error = e
            try:
                pos_vals = []
                for p in req:
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                        pos_vals.append(kwargs[p.name])
                inst = enc_cls(*pos_vals)
                print(f"[TriMF] Initialized {enc_cls.__name__} with signature positional: {[p.name for p in req]}")
                return inst
            except Exception:
                pass

    tries = []
    text_model = _pick_text_model()
    img_model = _pick_image_model()
    n_feats = extra.get("n_feats", None) or getattr(cfg, "structured_n_feats", None) or getattr(cfg, "n_feats", None)

    if is_text and text_model is not None:
        tries += [
            ("(model_name, d)",            lambda: enc_cls(text_model, d_model)),
            ("(model_name, d=d)",          lambda: enc_cls(text_model, d=d_model)),
            ("(cfg, model_name, d)",       lambda: enc_cls(cfg, text_model, d_model)),
            ("(model_name, cfg, d)",       lambda: enc_cls(text_model, cfg, d_model)),
            ("(cfg, d) [last resort]",     lambda: enc_cls(cfg, d_model)),
        ]
    elif is_image and img_model is not None:
        tries += [
            ("(model_name, d)",            lambda: enc_cls(img_model, d_model)),
            ("(model_name, d=d)",          lambda: enc_cls(img_model, d=d_model)),
            ("(cfg, model_name, d)",       lambda: enc_cls(cfg, img_model, d_model)),
            ("(cfg, d) [last resort]",     lambda: enc_cls(cfg, d_model)),
        ]
    else:
        if n_feats is not None:
            try_nf = int(n_feats)
        else:
            try_nf = None
        if try_nf is not None:
            tries += [
                ("(cfg, d, n_feats)",        lambda: enc_cls(cfg, d_model, try_nf)),
                ("(cfg, d=d, n_feats=nf)",   lambda: enc_cls(cfg, d=d_model, n_feats=try_nf)),
                ("(cfg, n_feats=nf, d=d)",   lambda: enc_cls(cfg, n_feats=try_nf, d=d_model)),
                ("(cfg, n_feats=nf)",        lambda: enc_cls(cfg, n_feats=try_nf)),
            ]
        tries += [
            ("(cfg, d)",                   lambda: enc_cls(cfg, d_model)),
            ("(cfg, d=d)",                 lambda: enc_cls(cfg, d=d_model)),
            ("(cfg)",                      lambda: enc_cls(cfg)),
            ("(d)",                        lambda: enc_cls(d_model)),
            ("()",                         lambda: enc_cls()),
        ]

    for name, fn in tries:
        try:
            inst = fn()
            print(f"[TriMF] Initialized {enc_cls.__name__} with {name}")
            return inst
        except TypeError as e:
            last_type_error = e
            continue

    hint = f" Last TypeError: {last_type_error}" if last_type_error is not None else ""
    if missing:
        raise TypeError(
            f"{enc_cls.__name__} missing required args {missing}. "
            f"Ensure cfg provides text_model_name/image_model_name (as appropriate), and labs n_feats." + hint
        )
    raise TypeError(f"Failed to initialize {enc_cls.__name__}." + hint)


class TriMF(nn.Module):
    def __init__(
        self,
        cfg: Any,
        num_labels: int = 25,
        d_model: int = 256,
        dropout: float = 0.2,
        n_feats: Optional[int] = None,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.d_model = int(d_model)

        enc_d = int(getattr(cfg, "d", getattr(cfg, "d_model", d_model)))

        nf = None
        if n_feats is not None:
            try:
                nf = int(n_feats)
            except Exception:
                nf = None

        if nf is None or nf <= 0:
            for k in ["structured_n_feats", "n_feats", "vocab_size", "structured_vocab_size"]:
                if hasattr(cfg, k):
                    try:
                        v = int(getattr(cfg, k))
                        if v > 0:
                            nf = v
                            break
                    except Exception:
                        pass

        extra_labs = {"n_feats": int(nf)} if (nf is not None and int(nf) > 0) else None

        self.lab_enc  = _instantiate_encoder(BEHRTLabEncoder,    cfg, enc_d, extra=extra_labs)
        self.note_enc = _instantiate_encoder(BioClinBERTEncoder, cfg, enc_d)
        self.img_enc  = _instantiate_encoder(ImageEncoder,       cfg, enc_d)

        self.proj_l = nn.Sequential(nn.LazyLinear(d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model))
        self.proj_n = nn.Sequential(nn.LazyLinear(d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model))
        self.proj_i = nn.Sequential(nn.LazyLinear(d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model))

        self.f_ln = PairFusion(d_model, dropout)
        self.f_li = PairFusion(d_model, dropout)
        self.f_ni = PairFusion(d_model, dropout)
        self.f_tri = TriFusion(d_model, dropout)

        self.head = nn.Linear(d_model, num_labels)

    def encode_labs(self, labs: Optional[Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
        if labs is None:
            return None

        if isinstance(labs, dict) and ("x" in labs) and (labs["x"] is not None):
            x = labs["x"].float()
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return self.proj_l(x)

        try:
            out = self.lab_enc(**labs)
        except TypeError:
            out = self.lab_enc(labs)
        emb = _pick_emb_from_encoder_out(out, attention_mask=labs.get("attention_mask", None))
        return self.proj_l(emb)

    def encode_notes(self, notes: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        if notes is None:
            return None

        kwargs = {k: v for k, v in notes.items() if k in ["input_ids", "attention_mask", "chunk_mask"]}
        raw = notes.get("raw_text", None)
        if raw is not None and isinstance(raw, list) and any(r is not None for r in raw):
            kwargs["text"] = raw

        try:
            out = self.note_enc(**kwargs)
        except TypeError:
            kwargs.pop("text", None)
            try:
                out = self.note_enc(**kwargs)
            except TypeError:
                out = self.note_enc(kwargs)

        t = _tensor_from_encoder_out(out)

        if t.ndim == 3:
            cm = notes.get("chunk_mask", None)
            if isinstance(cm, torch.Tensor) and cm.ndim == 2 and t.shape[1] == cm.shape[1]:
                cmf = cm.float().unsqueeze(-1)  # [B,C,1]
                denom = cmf.sum(dim=1).clamp_min(1.0)
                emb = (t * cmf).sum(dim=1) / denom
            else:
                am = kwargs.get("attention_mask", None)
                if isinstance(am, torch.Tensor) and am.ndim == 2 and am.shape[1] == t.shape[1]:
                    emb = _pool_tokens(t, attention_mask=am)
                else:
                    emb = t[:, 0, :]
        elif t.ndim == 2:
            emb = t
        elif t.ndim == 4:
            emb = t.flatten(1)
        else:
            raise ValueError(f"Unexpected notes tensor shape from encoder: {t.shape}")

        return self.proj_n(emb)

    def encode_images(self, images: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if images is None:
            return None
        try:
            out = self.img_enc(images)
        except TypeError:
            out = self.img_enc(pixel_values=images)
        emb = _pick_emb_from_encoder_out(out, attention_mask=None)
        return self.proj_i(emb)

    def forward(
        self,
        labs: Optional[Dict[str, torch.Tensor]] = None,
        notes: Optional[Dict[str, Any]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        zl = self.encode_labs(labs)
        zn = self.encode_notes(notes)
        zi = self.encode_images(images)

        bsz = None
        for z in [zl, zn, zi]:
            if z is not None:
                bsz = z.shape[0]
                break
        if bsz is None:
            raise ValueError("All modalities missing in forward().")

        device = (zl.device if zl is not None else (zn.device if zn is not None else zi.device))

        if zl is None:
            zl = torch.zeros((bsz, self.d_model), device=device)
        if zn is None:
            zn = torch.zeros((bsz, self.d_model), device=device)
        if zi is None:
            zi = torch.zeros((bsz, self.d_model), device=device)

        h_ln = self.f_ln(zl, zn)
        h_li = self.f_li(zl, zi)
        h_ni = self.f_ni(zn, zi)

        h = self.f_tri(h_ln, h_li, h_ni)
        logits = self.head(h)
        return {"logits": logits, "h": h, "h_ln": h_ln, "h_li": h_li, "h_ni": h_ni}

def _autocast_ctx(precision: str, device_type: str):
    precision = precision.lower()
    if device_type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", enabled=False)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_labels: int,
    precision: str,
    loss_fn: nn.Module,
    thresh: float = 0.5,
) -> Dict[str, Any]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    losses: List[float] = []

    for batch in loader:
        labs = batch["labs"]
        notes = batch["notes"]
        images = batch["images"]
        y = batch["y"].to(device)

        if labs is not None:
            labs = {k: v.to(device) for k, v in labs.items()}
        if notes is not None:
            notes = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in notes.items()}
        if images is not None:
            images = images.to(device)

        with _autocast_ctx(precision, device.type):
            out = model(labs=labs, notes=notes, images=images)
            logits = out["logits"]
            loss = loss_fn(logits, y)

        losses.append(float(loss.detach().cpu().item()))
        all_logits.append(logits.detach().cpu().float().numpy())
        all_y.append(y.detach().cpu().float().numpy())

    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, num_labels), dtype=np.float32)
    y_np = np.concatenate(all_y, axis=0) if all_y else np.zeros((0, num_labels), dtype=np.float32)

    metrics: Dict[str, Any] = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "thresh": float(thresh),
        "n": int(logits_np.shape[0]),
    }

    if logits_np.shape[0] == 0:
        # empty loader
        metrics.update({
            "auroc_micro": float("nan"),
            "auroc_macro": float("nan"),
            "auprc_micro": float("nan"),
            "auprc_macro": float("nan"),
            "auroc_per_label": [float("nan")] * num_labels,
            "auprc_per_label": [float("nan")] * num_labels,
            "precision_micro": float("nan"),
            "recall_micro": float("nan"),
            "f1_micro": float("nan"),
            "precision_macro": float("nan"),
            "recall_macro": float("nan"),
            "f1_macro": float("nan"),
            "precision_macro_pos": float("nan"),
            "recall_macro_pos": float("nan"),
            "f1_macro_pos": float("nan"),
            "precision_per_label": [float("nan")] * num_labels,
            "recall_per_label": [float("nan")] * num_labels,
            "f1_per_label": [float("nan")] * num_labels,
            "support_pos": [0] * num_labels,
            "support_neg": [0] * num_labels,
        })
        return metrics

    probs = 1.0 / (1.0 + np.exp(-logits_np))
    yb = (y_np >= 0.5).astype(np.int32)

    auroc_pl = [float("nan")] * num_labels
    auprc_pl = [float("nan")] * num_labels

    if _HAVE_SK:
        for j in range(num_labels):
            yj = yb[:, j]
            # undefined if only one class present
            if np.all(yj == 0) or np.all(yj == 1):
                continue
            try:
                auroc_pl[j] = float(roc_auc_score(yj, probs[:, j]))
                auprc_pl[j] = float(average_precision_score(yj, probs[:, j]))
            except Exception:
                continue

        valid_auroc = [v for v in auroc_pl if not np.isnan(v)]
        valid_auprc = [v for v in auprc_pl if not np.isnan(v)]
        metrics["auroc_macro"] = float(np.mean(valid_auroc)) if valid_auroc else float("nan")
        metrics["auprc_macro"] = float(np.mean(valid_auprc)) if valid_auprc else float("nan")
        metrics["auroc_macro_valid_n"] = int(len(valid_auroc))
        metrics["auprc_macro_valid_n"] = int(len(valid_auprc))

        try:
            metrics["auroc_micro"] = float(roc_auc_score(yb.ravel(), probs.ravel()))
        except Exception:
            metrics["auroc_micro"] = float("nan")
        try:
            metrics["auprc_micro"] = float(average_precision_score(yb.ravel(), probs.ravel()))
        except Exception:
            metrics["auprc_micro"] = float("nan")
    else:
        metrics["auroc_micro"] = float("nan")
        metrics["auroc_macro"] = float("nan")
        metrics["auprc_micro"] = float("nan")
        metrics["auprc_macro"] = float("nan")
        metrics["auroc_macro_valid_n"] = 0
        metrics["auprc_macro_valid_n"] = 0

    metrics["auroc_per_label"] = auroc_pl
    metrics["auprc_per_label"] = auprc_pl

    pred = (probs >= thresh).astype(np.int32)

    prec_pl: List[float] = []
    rec_pl: List[float] = []
    f1_pl: List[float] = []
    support_pos: List[int] = []
    support_neg: List[int] = []

    for j in range(num_labels):
        yt = yb[:, j]
        yp = pred[:, j]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())

        p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

        prec_pl.append(float(p))
        rec_pl.append(float(r))
        f1_pl.append(float(f1))
        support_pos.append(int(yt.sum()))
        support_neg.append(int((1 - yt).sum()))

    # micro
    TP = int(((pred == 1) & (yb == 1)).sum())
    FP = int(((pred == 1) & (yb == 0)).sum())
    FN = int(((pred == 0) & (yb == 1)).sum())

    p_micro = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    r_micro = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    f1_micro = (2.0 * p_micro * r_micro / (p_micro + r_micro)) if (p_micro + r_micro) > 0 else 0.0

    metrics["precision_micro"] = float(p_micro)
    metrics["recall_micro"] = float(r_micro)
    metrics["f1_micro"] = float(f1_micro)

    metrics["precision_macro"] = float(np.mean(prec_pl)) if prec_pl else float("nan")
    metrics["recall_macro"] = float(np.mean(rec_pl)) if rec_pl else float("nan")
    metrics["f1_macro"] = float(np.mean(f1_pl)) if f1_pl else float("nan")

    pos_idx = [j for j in range(num_labels) if support_pos[j] > 0]
    if pos_idx:
        metrics["precision_macro_pos"] = float(np.mean([prec_pl[j] for j in pos_idx]))
        metrics["recall_macro_pos"] = float(np.mean([rec_pl[j] for j in pos_idx]))
        metrics["f1_macro_pos"] = float(np.mean([f1_pl[j] for j in pos_idx]))
        metrics["macro_pos_n"] = int(len(pos_idx))
    else:
        metrics["precision_macro_pos"] = float("nan")
        metrics["recall_macro_pos"] = float("nan")
        metrics["f1_macro_pos"] = float("nan")
        metrics["macro_pos_n"] = 0

    metrics["precision_per_label"] = prec_pl
    metrics["recall_per_label"] = rec_pl
    metrics["f1_per_label"] = f1_pl
    metrics["support_pos"] = support_pos
    metrics["support_neg"] = support_neg

    return metrics


def compute_pos_weight_from_items(items: List[Dict[str, Any]], num_labels: int) -> torch.Tensor:
    ys = []
    for it in items:
        y = it.get("y", None)
        if y is None:
            continue
        if isinstance(y, torch.Tensor):
            ys.append(y.float().view(1, -1))
        elif isinstance(y, np.ndarray):
            ys.append(torch.from_numpy(y).float().view(1, -1))
        else:
            ys.append(torch.tensor(y, dtype=torch.float32).view(1, -1))

    if not ys:
        return torch.ones((num_labels,), dtype=torch.float32)

    Y = torch.cat(ys, dim=0)  # [N,K]
    pos = Y.sum(dim=0).double()
    total = Y.shape[0]
    neg = total - pos
    pos = pos.clamp_min(1.0)
    pw = (neg / pos).float()
    return pw



def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, epoch: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Any = None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt



def build_loaders(args, num_labels: int):
    train_items = load_split_items_auto(args, "train", num_labels=num_labels)
    val_items   = load_split_items_auto(args, "val",   num_labels=num_labels)
    test_items  = None
    try:
        test_items = load_split_items_auto(args, "test", num_labels=num_labels)
    except Exception:
        test_items = None

    image_tfm = build_image_transform(args.image_size, args.img_norm)
    img_roots: List[Union[str, Path]] = []
    if getattr(args, "image_root", ""):
        img_roots.append(Path(args.image_root))
    img_roots.append(Path(args.data_root))
    img_roots.append(Path(args.data_root).resolve().parent)
    def _collate(batch):
        return collate_multimodal(
            batch,
            num_labels=num_labels,
            image_tfm=image_tfm,
            image_roots=img_roots,
            skip_missing_images=getattr(args, "skip_missing_images", False),
        )
    train_ds = MultimodalCacheDataset(train_items, num_labels=num_labels)
    val_ds   = MultimodalCacheDataset(val_items,   num_labels=num_labels)
    test_ds  = MultimodalCacheDataset(test_items,  num_labels=num_labels) if test_items is not None else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=_collate,
        )
    def _cov(name, items):
        n = len(items)
        cL = sum(1 for it in items if it.get("labs") is not None)
        cN = sum(1 for it in items if it.get("notes") is not None)
        cI = sum(1 for it in items if it.get("image") is not None)
        print(f"[TriMF][{name}] coverage: labs={cL}/{n} notes={cN}/{n} images={cI}/{n}")
        if n > 0 and (cL + cN + cI) == 0:
            print(f"[TriMF][{name}] Example item keys:", list(items[0].keys()))
            print(f"[TriMF][{name}] Example item:", {k: type(items[0][k]).__name__ for k in items[0].keys()})

    _cov("train", train_items)
    _cov("val", val_items)
    if test_items is not None:
        _cov("test", test_items)
    if len(train_items) > 0 and all(it.get("labs") is None and it.get("notes") is None and it.get("image") is None for it in train_items[:200]):
        raise RuntimeError("All modalities are missing for train. Fix join/filter before training.")


    return train_loader, val_loader, test_loader


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pheno", choices=["pheno", "mort", "pe", "ph"], help="Task name (pheno expects 25 labels).")
    parser.add_argument("--data_root", type=str, required=True, help="Root dir containing cached splits.")
    parser.add_argument("--splits_json", type=str, default="splits.json", help="splits.json path (relative to data_root by default).")
    parser.add_argument("--structured_parquet", type=str, default="xehr_haru17_2h_76.parquet", help="Structured/Labs parquet (relative to data_root by default).")
    parser.add_argument("--notes_parquet", type=str, default="notes_fullstay_radiology_TEXTCHUNKS_11230.parquet", help="Notes parquet (relative to data_root by default).")
    parser.add_argument("--images_parquet", type=str, default="images.parquet", help="Images parquet (relative to data_root by default).")
    parser.add_argument("--labels_parquet", type=str, default="labels_pheno.parquet", help="Labels parquet (relative to data_root by default).")
    parser.add_argument("--id_col", type=str, default="stay_id", help="ID column name used across parquet files (default: stay_id).")
    parser.add_argument("--image_root", type=str, default="",
        help="Base directory used to resolve relative cxr_path (e.g., a dir that contains 'mimic-cxr-jpg'). "
             "If empty, will try data_root and data_root/.. automatically."
    )
    parser.add_argument(
            "--skip_missing_images",
        action="store_true",
        help="If an image file path doesn't exist, use a zero placeholder instead of crashing."
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device override (default: auto).")
    parser.add_argument("--ckpt_root", type=str, required=True, help="Where to save checkpoints/logs.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_labels", type=int, default=25)
    parser.add_argument("--n_feats", type=int, default=0, help="BEHRTLabEncoder vocab/feature count; 0=auto infer from train labs input_ids max+1.")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--img_norm", type=str, default="imagenet", choices=["imagenet", "none"])
    parser.add_argument("--thresh", type=float, default=0.5, help="Probability threshold for PR/F1 metrics (default 0.5).")
    parser.add_argument("--compute_pos_weight", action="store_true", help="Compute pos_weight from train set for BCEWithLogitsLoss.")
    parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume from.")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation (val/test) using --resume checkpoint.")
    args, unknown = parser.parse_known_args()

    set_seed(args.seed)

    cfg = None
    if _load_cfg is not None:
        try:
            cfg = _load_cfg()
        except TypeError:
            try:
                cfg = _load_cfg(task=args.task, data_root=args.data_root, ckpt_root=args.ckpt_root)
            except Exception:
                try:
                    cfg = _load_cfg(args)
                except Exception:
                    cfg = None

    if cfg is None:
        cfg = _CFG if _CFG is not None else SimpleNamespace()

    if _apply_cli_overrides is not None:
        try:
            cfg = _apply_cli_overrides(cfg, args, unknown)
        except Exception:
            try:
                _apply_cli_overrides(cfg, args)
            except Exception:
                pass

    device = _DEVICE if _DEVICE is not None else _default_device()
    if args.device != 'auto':
        device = torch.device(args.device)
    elif device.type == 'cpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    print(f"Device: {device} | precision={args.precision} | task={args.task} | num_labels={args.num_labels}")

    train_loader, val_loader, test_loader = build_loaders(args, num_labels=args.num_labels)
    print(f"Loaded splits: train={len(train_loader.dataset)} | val={len(val_loader.dataset)}" + (f" | test={len(test_loader.dataset)}" if test_loader else ""))

    if getattr(args, "n_feats", 0) <= 0:
        cfg_nf = None
        cfg_nf_src = None
        for k in ["structured_n_feats", "n_feats", "vocab_size", "structured_vocab_size"]:
            if hasattr(cfg, k):
                try:
                    v = int(getattr(cfg, k))
                    if v > 0:
                        cfg_nf = v
                        cfg_nf_src = k
                        break
                except Exception:
                    pass

        if cfg_nf is not None and cfg_nf > 0:
            args.n_feats = int(cfg_nf)
            print(f"Using n_feats={args.n_feats} from cfg.{cfg_nf_src}.")
        else:
            try:
                items0: List[Dict[str, Any]] = []
                items0 += list(getattr(train_loader.dataset, "items", []))
                items0 += list(getattr(val_loader.dataset, "items", []))
                if test_loader is not None:
                    items0 += list(getattr(test_loader.dataset, "items", []))
                args.n_feats = infer_n_feats_from_items(items0, fallback=cfg_nf)
                print(f"Inferred n_feats={args.n_feats} from labs input_ids (train/val" + ("/test" if test_loader is not None else "") + ").")
            except Exception as e:
                args.n_feats = int(cfg_nf) if (cfg_nf is not None and int(cfg_nf) > 0) else 1
                print(f"[WARN] Could not infer n_feats automatically ({e}); using n_feats={args.n_feats}.")

    model = TriMF(cfg, num_labels=args.num_labels, d_model=args.d_model, dropout=args.dropout, n_feats=args.n_feats).to(device)

    pos_weight = None
    if args.compute_pos_weight:
        print("Computing pos_weight from train labels (no DataLoader iteration)...")
        train_items_raw = getattr(train_loader.dataset, "items", [])
        pos_weight = compute_pos_weight_from_items(train_items_raw, num_labels=args.num_labels).to(device)
        print("pos_weight (first 10):", pos_weight[:10].detach().cpu().numpy().round(3))

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    ckpt_root = Path(args.ckpt_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_root / "trimf_best.pt"
    last_path = ckpt_root / "trimf_last.pt"
    metrics_path = ckpt_root / "trimf_metrics.jsonl"

    start_epoch = 0
    best_metric = -1e9

    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model, optimizer if not args.eval_only else None, scheduler if not args.eval_only else None)
        start_epoch = int(ckpt.get("epoch", 0)) + (0 if args.eval_only else 1)
        best_metric = float(ckpt.get("best_metric", best_metric))
        print(f"Loaded checkpoint: {args.resume} | start_epoch={start_epoch} | best_metric={best_metric}")

    if args.eval_only:
        assert args.resume, "--eval_only requires --resume"
        val_metrics = evaluate(model, val_loader, device, args.num_labels, args.precision, loss_fn, thresh=args.thresh)
        print("VAL:", val_metrics)
        if test_loader is not None:
            test_metrics = evaluate(model, test_loader, device, args.num_labels, args.precision, loss_fn, thresh=args.thresh)
            print("TEST:", test_metrics)
        return

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.precision == "fp16"))

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = []

        for it, batch in enumerate(train_loader):
            labs = batch["labs"]
            notes = batch["notes"]
            images = batch["images"]
            y = batch["y"].to(device)

            if labs is not None:
                labs = {k: v.to(device) for k, v in labs.items()}
            if notes is not None:
                notes = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in notes.items()}
            if images is not None:
                images = images.to(device)

            optimizer.zero_grad(set_to_none=True)

            with _autocast_ctx(args.precision, device.type):
                out = model(labs=labs, notes=notes, images=images)
                logits = out["logits"]
                loss = loss_fn(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running.append(float(loss.detach().cpu().item()))
            global_step += 1

            if args.log_every > 0 and (global_step % args.log_every == 0):
                avg_loss = float(np.mean(running[-args.log_every:])) if len(running) >= args.log_every else float(np.mean(running))
                lr = optimizer.param_groups[0]["lr"]
                print(f"[epoch {epoch:03d} step {global_step:06d}] train_loss={avg_loss:.4f} lr={lr:g}")

        train_loss = float(np.mean(running)) if running else float("nan")
        val_metrics = evaluate(model, val_loader, device, args.num_labels, args.precision, loss_fn, thresh=args.thresh)
        key = "auroc_macro" if ("auroc_macro" in val_metrics and not np.isnan(val_metrics["auroc_macro"])) else "loss"
        metric_val = val_metrics[key] if key != "loss" else -val_metrics["loss"]  # higher better for scheduler
        scheduler.step(metric_val)

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "lr": lr,
            "time_sec": dt,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")

        print(f"Epoch {epoch:03d} done in {dt:.1f}s | train_loss={train_loss:.4f} | val={val_metrics}")

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_metric)

        current = val_metrics.get("auroc_macro", float("nan"))
        if np.isnan(current):
            current = -val_metrics["loss"]
        if current > best_metric:
            best_metric = float(current)
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_metric)
            print(f"  New best: {best_metric:.4f} saved to {best_path}")

    if test_loader is not None:
        load_checkpoint(best_path, model)
        test_metrics = evaluate(model, test_loader, device, args.num_labels, args.precision, loss_fn, thresh=args.thresh)
        print("TEST (best):", test_metrics)


if __name__ == "__main__":
    main()
