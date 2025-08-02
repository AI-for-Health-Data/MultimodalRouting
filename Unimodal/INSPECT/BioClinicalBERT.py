import sys
sys.argv = sys.argv[:1] 
import argparse
import gzip
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, logging as tf_logging
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split

tf_logging.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")

os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"]         = CACHE_DIR

CONFIG = {
    "data_file": "radiology_impressions_with_all_labels.csv.gz",
    "patient_id_col": "person_id",
    "text_col": "impression_text",
    "label_cols": [
        "pe_positive_nlp",
        "1_month_mortality",
        "1_month_readmission",
        "12_month_PH",
    ],
    "model_name": "emilyalsentzer/Bio_ClinicalBERT",
    "max_length": 512,
    "batch_size": 16,
    "epochs": 10,
    "lr": 2e-5,
    "patience": 3,
    "test_size": 0.15,
    "val_size": 0.05,
    "aggregation": "mean",  # choose from "mean", "max", "sum"
    "demographic_cols": {
        "age": "year_of_birth"
    },
}

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()

class BioClinicalBERT(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:,0,:]  # CLS token

class PatientNotesDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.emb = torch.tensor(embeddings, dtype=torch.float32)
        self.lab = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        return self.emb[i], self.lab[i]

class MultiTaskClassifier(nn.Module):
    def __init__(self, input_size, num_labels, hidden_size=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, x):
        return self.net(x)

def assign_age_bucket(age: int) -> str:
    if 18 <= age <= 29: return "18-29"
    if 30 <= age <= 49: return "30-49"
    if 50 <= age <= 69: return "50-69"
    if 70 <= age <= 89: return "70-89"
    return "90+"

def stratified_split(X, y, test_size, val_size):
    while True:
        X_tv, y_tv, X_te, y_te = iterative_train_test_split(X, y, test_size=test_size)
        val_frac = val_size / (1 - test_size)
        X_tr, y_tr, X_va, y_va = iterative_train_test_split(X_tv, y_tv, test_size=val_frac)
        def good(Y):
            return all(set(np.unique(Y[:,i])) == {0,1} for i in range(Y.shape[1]))
        if good(y_tr) and good(y_va) and good(y_te):
            return X_tr, X_va, X_te, y_tr, y_va, y_te


def generate_embeddings(df, cfg, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], cache_dir=CACHE_DIR)
    model = BioClinicalBERT(cfg["model_name"]).to(device)
    model.eval()

    emb_map = {}
    groups = df.groupby(cfg["patient_id_col"])[cfg["text_col"]].apply(list)
    logging.info(f"Generating embeddings for {len(groups)} patients…")

    with torch.no_grad(), autocast():
        for pid, notes in tqdm(groups.items(), desc="Patients"):
            chunks = []
            for note in notes:
                toks = tokenizer(note, return_tensors="pt", add_special_tokens=True, truncation=False)["input_ids"][0]
                for i in range(0, toks.size(0), cfg["max_length"]):
                    chunk = toks[i:i+cfg["max_length"]].unsqueeze(0).to(device)
                    attn = torch.ones_like(chunk)
                    emb = model(chunk, attn)
                    chunks.append(emb.cpu().numpy()[0])
            if not chunks:
                emb_map[pid] = np.zeros(model.bert.config.hidden_size)
            else:
                arr = np.stack(chunks, axis=0)
                if cfg["aggregation"] == "mean":
                    emb_map[pid] = arr.mean(0)
                elif cfg["aggregation"] == "max":
                    emb_map[pid] = arr.max(0)
                else:
                    emb_map[pid] = arr.sum(0)
    return emb_map


def main(cfg, args):
    set_seed(42)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")
    
    start = time.time()
    with gzip.open(cfg["data_file"], "rt") as f:
        df = pd.read_csv(f, low_memory=False)
    df[cfg["text_col"]] = df[cfg["text_col"]].fillna("")
    df = df.dropna(subset=cfg["label_cols"])
    df = df[df[cfg["text_col"]].str.strip() != ""]
    logging.info(f"Loaded {len(df):,} rows; {df[cfg['patient_id_col']].nunique():,} patients")

    sampled_pids = (
        df[cfg["patient_id_col"]]
        .drop_duplicates()
        .sample(n=10, random_state=42)
    )
    df = df[df[cfg["patient_id_col"]].isin(sampled_pids)]
    logging.info(f"Subsampled to {len(df):,} rows; {df[cfg['patient_id_col']].nunique():,} patients")

    # load data
    #with gzip.open(cfg["data_file"], "rt") as f:
    #    df = pd.read_csv(f, low_memory=False)
    #df[cfg["text_col"]] = df[cfg["text_col"]].fillna("")
    #df = df.dropna(subset=cfg["label_cols"])
    #df = df[df[cfg["text_col"]].str.strip() != ""]
    #logging.info(f"Loaded {len(df):,} rows; {df[cfg['patient_id_col']].nunique():,} patients")

    emb_map = generate_embeddings(df, cfg, device)
    df["emb"] = df[cfg["patient_id_col"]].map(emb_map)

    df_u = df.drop_duplicates(cfg["patient_id_col"]).copy()
    df_u["emb"] = df_u[cfg["patient_id_col"]].map(emb_map)
    df_u["age"] = args.current_year - df_u[cfg["demographic_cols"]["age"]]
    df_u["age_bucket"] = df_u["age"].apply(assign_age_bucket)

    df_u[cfg["label_cols"]] = (
        df_u[cfg["label_cols"]]
        .apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    )

    X = np.stack(df_u["emb"].values)
    y = df_u[cfg["label_cols"]].values

    X_tr, X_va, X_te, y_tr, y_va, y_te = stratified_split(
        X, y, cfg["test_size"], cfg["val_size"]
    )
    logging.info(f"Splits → train {len(X_tr):,}, val {len(X_va):,}, test {len(X_te):,}")

    bs = cfg["batch_size"]
    train_dl = DataLoader(PatientNotesDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
    val_dl   = DataLoader(PatientNotesDataset(X_va, y_va), batch_size=bs)
    test_dl  = DataLoader(PatientNotesDataset(X_te, y_te), batch_size=bs)

    neg_pos = []
    for i in range(y_tr.shape[1]):
        neg = (y_tr[:,i] == 0).sum()
        pos = (y_tr[:,i] == 1).sum()
        neg_pos.append(neg / pos if pos>0 else 1.0)
    pos_weight = torch.tensor(neg_pos, device=device, dtype=torch.float32)

    model = MultiTaskClassifier(X_tr.shape[1], y_tr.shape[1]).to(device)
    model.eval()
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25, pos_weight=pos_weight)
    opt = AdamW(model.parameters(), lr=cfg["lr"])

    best_val = float("inf"); epochs_no_improve = 0
    best_path = "best_model.pt"

    logging.info("Starting training…")
    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        tloss=0
        for xb,yb in train_dl:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            l = loss_fn(logits, yb)
            l.backward(); opt.step()
            tloss += l.item()
        tloss /= len(train_dl)

        model.eval()
        vloss=0
        with torch.no_grad():
            for xb,yb in val_dl:
                xb,yb=xb.to(device),yb.to(device)
                vloss += loss_fn(model(xb), yb).item()
        vloss /= len(val_dl)

        logging.info(f"Epoch {epoch}/{cfg['epochs']}  train={tloss:.4f}  val={vloss:.4f}")
        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), best_path)
            epochs_no_improve=0
        else:
            epochs_no_improve+=1
            if epochs_no_improve>=cfg["patience"]:
                logging.info("Early stopping.")
                break

    model.load_state_dict(torch.load(best_path))
    model.eval()
    all_lbl, all_prob = [], []
    with torch.no_grad():
        for xb,yb in test_dl:
            xb=xb.to(device)
            prob = torch.sigmoid(model(xb)).cpu().numpy()
            all_prob.append(prob)
            all_lbl.append(yb.numpy())
    all_prob = np.vstack(all_prob)
    all_lbl  = np.vstack(all_lbl)
    all_pred = (all_prob>0.5).astype(int)

    logging.info("Test metrics:")
    for i,label in enumerate(cfg["label_cols"]):
        y_t,y_p,y_pred = all_lbl[:,i], all_prob[:,i], all_pred[:,i]
        if len(np.unique(y_t))<2:
            logging.info(f"  {label}: only one class; skipping")
            continue
        auc  = safe_metric(roc_auc_score, y_t, y_p)
        aupr = safe_metric(average_precision_score, y_t, y_p)
        rec  = safe_metric(recall_score, y_t, y_pred, zero_division=0)
        f1   = safe_metric(f1_score, y_t, y_pred, zero_division=0)
        logging.info(
            f"  {label:<20} AUROC={auc:.3f}  AUPRC={aupr:.3f}  "
            f"Recall={rec:.3f}  F1={f1:.3f}"
        )

    logging.info(f"Done in {(time.time()-start)/60:.1f} min")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--current-year", type=int, default=datetime.now().year)
    p.add_argument("--device", choices=["cpu","cuda"])
    args, _ = p.parse_known_args()   
    main(CONFIG, args)
