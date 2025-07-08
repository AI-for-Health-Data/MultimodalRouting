# FAME++ – Multimodal Routing for Fair ICU Outcome Prediction


| Modality                 | Encoder                | Notes                           |
| ------------------------ | ---------------------- | ------------------------------- |
| Structured labs & vitals | **BEHRTLabEncoder**    | 24 h time‑series → \[CLS] token |
| Free‑text notes          | **BioClinBERTEncoder** | concatenated admission notes    |
| Chest X‑ray image        | **??**    | --     |

The model creates **7 interaction routes** (3 × uni, 3 × bi, 1 × tri) and fuses
them with a deterministic, performance‑based router.

---


## Repository layout

```
MultimodalRouting/
├── train_fame.py         
├── routing.py            
│
├── models/
│   ├── encoders.py        # BEHRTLabEncoder, BioClinBERTEncoder, ImageEncoderModel(Zhongjie)
│   └── routes.py          # tiny MLP heads per interaction route
│
├── data/
│   └── icustay_dataset.py # loads MIMIC‑IV + INSPECT samples as PyTorch Dataset
│
├── utils/
│   └── fairness.py        # Error‑Distribution Disparity Index (EDDI)
└── cfgs/                  
```

---

## How the model trains 

1. **Stage 1 (uni)** – train only unimodal heads until they converge.
2. **Stage 2 (bi)** – freeze uni‑modal heads, train bi‑modal heads on the *residual*.
3. **Stage 3 (tri)** – freeze uni+bi heads, train the tri‑modal head.
4. After every batch we feed an **EMA of per‑route BCE losses** into
   `MMRouting`, which calculates fresh weights for the next forward pass.
5. Total loss = BCE + λ·EDDI (EDDI promotes fairness across sensitive groups).

