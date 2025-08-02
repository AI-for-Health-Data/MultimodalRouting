import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm

class FinalStructuredDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, compression='gzip', low_memory=False)
        df.fillna('missing', inplace=True)
        self.df = df

        # Labels
        self.labels = [
            'pe_positive_nlp',
            '1_month_mortality',
            '1_month_readmission',
            '12_month_PH'
        ]
        for c in self.labels:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

        # Factorize procedure, measurement, and drug concept names
        self.proc_col = 'procedure_concept_name'
        self.meas_col = 'measurement_concept_name'
        self.drug_col = 'drug_concept_name'

        for col in [self.proc_col, self.meas_col, self.drug_col]:
            df[col] = df[col].astype(str)
            df[f'{col}_id'] = pd.factorize(df[col])[0]

        self.proc_ids = df[f'{self.proc_col}_id'].values.astype(np.int64)
        self.meas_ids = df[f'{self.meas_col}_id'].values.astype(np.int64)
        self.drug_ids = df[f'{self.drug_col}_id'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        proc = torch.tensor(self.proc_ids[idx],   dtype=torch.long)
        meas = torch.tensor(self.meas_ids[idx],   dtype=torch.long)
        drug = torch.tensor(self.drug_ids[idx],   dtype=torch.long)
        lbls = torch.tensor([self.df.iloc[idx][c] for c in self.labels],
                             dtype=torch.float32)
        return proc, meas, drug, lbls

class CombinedModel(nn.Module):
    def __init__(self,
                 num_proc_codes,
                 num_meas_codes,
                 num_drug_codes,
                 hidden=128):
        super().__init__()
        # Embeddings
        self.proc_emb = nn.Embedding(num_proc_codes, hidden)
        self.meas_emb = nn.Embedding(num_meas_codes, hidden)
        self.drug_emb = nn.Embedding(num_drug_codes, hidden)
        self.fuse = nn.Linear(hidden * 3, hidden)
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({
            'pe':     nn.Linear(hidden, 1),
            'mort1m': nn.Linear(hidden, 1),
            'read1m': nn.Linear(hidden, 1),
            'ph12m':  nn.Linear(hidden, 1),
        })

    def forward(self, proc, meas, drug):
        h_proc = self.proc_emb(proc)
        h_meas = self.meas_emb(meas)
        h_drug = self.drug_emb(drug)
        # Since each is single embedding, no pooling needed
        h = torch.cat([h_proc, h_meas, h_drug], dim=1)
        h = torch.relu(self.fuse(h))
        h = self.drop(h)
        return {k: head(h).squeeze(-1) for k, head in self.heads.items()}


def compute_pos_weights(loader):
    all_lbl = torch.cat([b[-1] for b in loader], dim=0)
    w = []
    for i in range(all_lbl.shape[1]):
        pos = (all_lbl[:, i] == 1).sum().item()
        neg = (all_lbl[:, i] == 0).sum().item()
        w.append(neg / (pos + 1e-6))
    return w


def train_model(model, train_loader, val_loader, device,
                epochs=5, lr=1e-3, wd=1e-2, patience=3):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)
    pw = compute_pos_weights(train_loader)
    losses = {i: nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw[i], device=device))
              for i in range(4)}
    best_val, no_improve = math.inf, 0
    for epoch in range(1, epochs+1):
        # Train
        model.train()
        train_losses = []
        for proc, meas, drug, lbl in tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False):
            proc, meas, drug, lbl = proc.to(device), meas.to(device), drug.to(device), lbl.to(device)
            optimizer.zero_grad()
            outs = model(proc, meas, drug)
            loss = sum(losses[i](outs[k], lbl[:, i]) for i, k in enumerate(['pe','mort1m','read1m','ph12m']))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train = np.mean(train_losses)
        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for proc, meas, drug, lbl in tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False):
                proc, meas, drug, lbl = proc.to(device), meas.to(device), drug.to(device), lbl.to(device)
                outs = model(proc, meas, drug)
                loss = sum(losses[i](outs[k], lbl[:, i]) for i, k in enumerate(['pe','mort1m','read1m','ph12m']))
                val_losses.append(loss.item())
        avg_val = np.mean(val_losses)
        print(f"Epoch {epoch}  Train {avg_train:.4f}  Val {avg_val:.4f}")
        scheduler.step(avg_val)
        if avg_val < best_val:
            best_val, no_improve = avg_val, 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break


def evaluate_model(model, loader, device, thresh=0.5):
    model.to(device).eval()
    all_lbl, all_proba, all_pred = [], {k:[] for k in model.heads}, {k:[] for k in model.heads}
    with torch.no_grad():
        for proc, meas, drug, lbl in tqdm(loader, desc="Evaluating", leave=False):
            proc, meas, drug, lbl = proc.to(device), meas.to(device), drug.to(device), lbl.to(device)
            outs = model(proc, meas, drug)
            for k, logits in outs.items():
                proba = torch.sigmoid(logits).cpu().numpy()
                pred  = (proba>thresh).astype(int)
                all_proba[k].append(proba)
                all_pred[k].append(pred)
            all_lbl.append(lbl.cpu().numpy())
    for k in all_proba:
        all_proba[k] = np.concatenate(all_proba[k])
        all_pred[k]  = np.concatenate(all_pred[k])
    all_lbl = np.vstack(all_lbl)
    metrics = {}
    for i,k in enumerate(['pe','mort1m','read1m','ph12m']):
        y_true, y_proba, y_pred = all_lbl[:,i], all_proba[k], all_pred[k]
        metrics[k] = {
            'auroc': roc_auc_score(y_true, y_proba),
            'auprc': average_precision_score(y_true, y_proba),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
    return metrics

def main():
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Subset
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    torch.manual_seed(42)
    np.random.seed(42)

    CSV = "structured_ehr_all.csv.gz"
    ds  = FinalStructuredDataset(CSV)
    N   = len(ds)
    print(f"Dataset size: {N:,}")

    lab_df = pd.read_csv(CSV, compression="gzip", usecols=ds.labels, low_memory=False)
    for c in ds.labels:
        lab_df[c] = pd.to_numeric(lab_df[c], errors="coerce").fillna(0).astype(int)
    lab_vals = lab_df.values

    # 80/5/15 multilabel‐stratified split
    m1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(m1.split(np.zeros(N), lab_vals))

    val_frac = 0.05 / 0.85
    m2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    train_idx_rel, val_idx_rel = next(
        m2.split(np.zeros(len(train_val_idx)), lab_vals[train_val_idx])
    )
    train_idx = train_val_idx[train_idx_rel]
    val_idx   = train_val_idx[val_idx_rel]

    # Fast DataLoaders
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(ds, test_idx),
        batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True
    )

    num_proc = int(ds.proc_ids.max()) + 1
    num_meas = int(ds.meas_ids.max()) + 1
    num_drug = int(ds.drug_ids.max()) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CombinedModel(num_proc, num_meas, num_drug, hidden=128)

    train_model(model, train_loader, val_loader, device)

    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    metrics = evaluate_model(model, test_loader, device)

    print("\n=== Test Metrics ===")
    for task, m in metrics.items():
        print(f"{task}: AUROC {m['auroc']:.3f}, AUPRC {m['auprc']:.3f}, "
              f"Rec {m['recall']:.3f}, F1 {m['f1']:.3f}")

if __name__ == "__main__":
    main()

