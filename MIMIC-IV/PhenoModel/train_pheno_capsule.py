import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from encoders import build_encoders, EncoderConfig 
from routing_and_heads import ConcatMLPFusion       
from PhenoModel.capsule_model_pheno import PhenoCapsuleHead


def build_fuser(d: int, out_dim: int) -> nn.Module:
    """
    Simple concat + MLP: [zL | zN | zI] -> fused z
    Expect each encoder to output dim=d.
    """
    hidden = 2 * out_dim
    return nn.Sequential(
        nn.LayerNorm(3 * d),
        nn.Linear(3 * d, hidden),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, out_dim),
    )


def train_one_epoch(model, loss_fn, optimizers, data_loader, device):
    model.train()
    for batch in data_loader:
        xL, mL, notes, imgs, y = batch
        y = y.to(device)

        behrt, bbert, imgenc, fuser, head = model["behrt"], model["bbert"], model["imgenc"], model["fuser"], model["head"]

        with torch.no_grad():  # freeze encoders if you wish
            zL = behrt(xL.to(device), mask=mL.to(device) if mL is not None else None)     
            zN = bbert(notes)                                                               
            zI = imgenc(imgs)                                                              

        z = torch.cat([zL, zN, zI], dim=-1)                                                
        z = fuser(z)                                                                        

        out = head(z)                                                                       
        logits = out["logits"]                                                            

        loss = loss_fn(logits, y.float())

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--d_fuse", type=int, default=256)
    parser.add_argument("--n_primary", type=int, default=32)
    parser.add_argument("--d_primary", type=int, default=16)
    parser.add_argument("--d_class", type=int, default=16)
    parser.add_argument("--routing_iters", type=int, default=3)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--freeze_encoders", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = EncoderConfig(d=args.d)
    behrt, bbert, imgenc = build_encoders(cfg, device=device)

    # Fusion (concat+MLP) 
    fuser = build_fuser(d=cfg.d, out_dim=args.d_fuse).to(device)

    # Capsule phenotyping head 
    head = PhenoCapsuleHead(
        d_in=args.d_fuse,
        n_primary=args.n_primary,
        d_primary=args.d_primary,
        n_classes=args.num_classes,
        d_class=args.d_class,
        routing_iters=args.routing_iters,
        act_type="EM",
        dp=0.1,
        use_squash_primary=False,
    ).to(device)

    # Loss / Opt
    loss_fn = nn.BCEWithLogitsLoss()
    optimizers = [
        optim.AdamW(fuser.parameters(), lr=args.lr),
        optim.AdamW(head.parameters(), lr=args.lr),
    ]

if __name__ == "__main__":
    main()
