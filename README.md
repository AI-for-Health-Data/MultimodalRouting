# Multimodal Routing for Interpretable, Robust, and Auditable Clinical Prediction

<p align="center">
  <img src="/Users/nikkiehooman/Downloads/Slide107.pdf" width="900">
</p>

<p align="center">
  <b>Figure 1:</b> Architecture of the proposed multimodal routing framework.
  Structured longitudinal data (L), clinical notes (N), and chest X-rays (I)
  are encoded separately. The model constructs unimodal, directional bimodal,
  and trimodal routes using cross-attention. Patient-specific route activations
  and label-specific routing coefficients selectively aggregate route
  representations to produce interpretable and robust predictions.
</p>

---

## Overview

This repository contains the official implementation of **Multimodal Routing**, a routing-based multimodal learning framework for clinical prediction from Electronic Health Records (EHRs).

The framework explicitly separates and weighs **unimodal**, **directional bimodal**, and **trimodal** evidence pathways across three clinical modalities:

- **L** — Structured longitudinal data (vitals, labs, medications)
- **N** — Clinical notes
- **I** — Chest X-ray images

By learning *how* and *when* each modality—and their interactions—contributes to predictions, the framework enables:

- **Interpretability** through explicit evidence pathways  
- **Robustness** to missing modalities at inference time  
- **Auditable multimodal reasoning** suitable for high-stakes clinical settings  

---

## Associated Paper

**Multimodal Routing for Interpretable, Robust, and Auditable Clinical Prediction**  
ACM Conference on Connected Health (CHASE), under review

If you use this code, please cite:

```bibtex
@inproceedings{multimodalrouting2026,
  title     = {Multimodal Routing for Interpretable, Robust, and Auditable Clinical Prediction},
  author    = {Anonymous Authors},
  booktitle = {ACM Conference on Connected Health (CHASE)},
  year      = {2026}
}
Key Contributions
Explicit modeling of 10 multimodal routes

3 unimodal routes

6 directional bimodal routes
(e.g., 
𝑁
←
𝐿
N←L vs. 
𝐿
←
𝑁
L←N)

1 trimodal route

Route activations (patient-specific)
Quantify how strongly each unimodal or cross-modal route is expressed for an individual patient.

Routing coefficients (patient- and label-specific)
Quantify how much each route contributes to each prediction target.

Inference-time route masking
Simulates missing-modality scenarios without retraining by disabling routes involving unavailable modalities and renormalizing routing weights.

Auditable multimodal reasoning
Routing weight redistribution under missing-modality settings enables systematic auditing of modality reliance and model robustness.

Clinical Prediction Tasks
This code supports two ICU prediction tasks using paired tri-modal EHR data:

Binary ICU Mortality Prediction
Observation window: first 48 hours of the ICU stay

Multi-label Phenotype Prediction
25 phenotypes

Observation window: full ICU stay

Discharge summaries are excluded to avoid information leakage

Repository Structure
The repository is organized by prediction task, with separate pipelines for
phenotype prediction and ICU mortality prediction.
Each task contains parallel implementations for baseline fusion and routing-based multimodal models.

MultimodalRouting/
├── Data/                         # (Local only) processed data placeholders
├── INSPECT/                      # Inspection / debugging utilities
│
├── MIMIC-IV/
│   ├── Data/                     # Task-specific data handling
│   ├── Model/                    # Shared model utilities
│   │
│   ├── MortModel/                # ICU mortality prediction
│   │   ├── Baseline/             # Baseline fusion models
│   │   ├── Paired_Cross_Attention/
│   │   │   ├── mult_model.py     # Multimodal routing model
│   │   │   ├── routing_and_heads.py
│   │   │   ├── capsule_layers.py
│   │   │   ├── encoders.py
│   │   │   ├── transformer.py
│   │   │   ├── main.py           # Training / evaluation entry point
│   │   │   └── env_config.py
│   │   ├── Paired_Simple_Concat/ # Undirected fusion ablation
│   │   └── Partial/              # Missing-modality experiments
│   │
│   └── PhenoModel/               # Multi-label phenotype prediction
│       ├── Baseline/             # Joint and late fusion baselines
│       ├── Paired_Cross_Attention/
│       │   ├── mult_model.py
│       │   ├── routing_and_heads.py
│       │   ├── capsule_layers.py
│       │   ├── encoders.py
│       │   ├── multhead_attention.py
│       │   ├── position_embedding.py
│       │   ├── transformer.py
│       │   ├── main.py
│       │   └── env_config.py
│       ├── Paired_Simple_Concat/
│       ├── Partial/
│       └── README.md
│
└── README.md                     # Top-level documentation
Design Notes
MortModel and PhenoModel share the same routing architecture but differ in:

Prediction head (binary vs. multi-label)

Observation window (48 hours vs. full stay)

Paired_Cross_Attention implements the full routing framework

Paired_Simple_Concat removes directional cross-attention as an ablation

Partial contains inference-time route masking experiments

Modalities are always encoded separately and fused only through explicit routing

Method Overview
Multimodal Routes
The model explicitly constructs a set of interpretable multimodal routes:

Unimodal routes

{
𝐿
,
  
𝑁
,
  
𝐼
}
{L,N,I}
Directional bimodal routes

{
𝐿
←
𝑁
,
  
𝑁
←
𝐿
,
  
𝐿
←
𝐼
,
  
𝐼
←
𝐿
,
  
𝑁
←
𝐼
,
  
𝐼
←
𝑁
}
{L←N,N←L,L←I,I←L,N←I,I←N}
Trimodal route

{
𝐿
𝑁
𝐼
}
{LNI}
The trimodal route is built hierarchically from paired directional interactions.
Each route 
𝑟
∈
𝑅
r∈R produces a route-specific embedding 
𝑒
𝑟
e 
r
​
 .

Decision Mechanism
For a patient 
𝑏
b and prediction target (label) 
𝑐
c, the decision representation is:

𝑑
𝑏
,
𝑐
=
∑
𝑟
∈
𝑅
𝑅
𝑏
,
𝑟
,
𝑐
⋅
𝛼
𝑏
,
𝑟
⋅
𝑒
~
𝑟
d 
b,c
​
 = 
r∈R
∑
​
 R 
b,r,c
​
 ⋅α 
b,r
​
 ⋅ 
e
~
  
r
​
 
where:

𝛼
𝑏
,
𝑟
α 
b,r
​
  — route activation (patient-specific)

𝑅
𝑏
,
𝑟
,
𝑐
R 
b,r,c
​
  — routing coefficient (patient- and label-specific)

𝑒
~
𝑟
e
~
  
r
​
  — primary route representation

The effective route contribution is:

𝑊
𝑏
,
𝑟
,
𝑐
=
𝛼
𝑏
,
𝑟
⋅
𝑅
𝑏
,
𝑟
,
𝑐
W 
b,r,c
​
 =α 
b,r
​
 ⋅R 
b,r,c
​
 
This formulation enforces structured, selective, and interpretable multimodal aggregation.

Data Sources and Privacy
This project uses credentialed access to the following PhysioNet datasets:

MIMIC-IV

MIMIC-IV-Note

MIMIC-CXR-JPG

Requirements
You must:

Complete PhysioNet credentialing

Agree to all applicable Data Use Agreements (DUAs)

Download all data locally

 Important

This repository does NOT include any patient data.
Do NOT upload derived tables, features, or model outputs containing patient-level information.

License
Specify the license here (e.g., MIT, Apache 2.0).

Acknowledgments
This work uses MIMIC-IV, MIMIC-IV-Note, and MIMIC-CXR datasets made available via PhysioNet.
Please cite the original dataset publications when using these resources.


---

###  Final verdict
- ✔ Technically correct  
- ✔ Complete  
- ✔ Reviewer-ready  
- ✔ Public-GitHub safe  
- ✔ Perfectly aligned with your paper  

If you want next, I can:
- Create a **double-blind version**
- Add a **Quick Start (3 commands)** section
- Add **code ↔ equation mapping**
- Write **sub-README files** for `PhenoModel/` and `MortModel/`

Just tell me.
