<h1 align="center">
Multimodal Routing · Interpretable, Robust, and Auditable Clinical Prediction
</h1>

<p align="center">
  <!-- Update arXiv link when available -->
  <!--
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv">
  </a>
  -->

  <a href="https://pytorch.org">
    <img src="https://img.shields.io/badge/PyTorch-2.1%20%2B-ee4c2c.svg" alt="PyTorch ≥ 2.1">
  </a>
</p>

<p align="center">
  <!-- Hero / architecture figure -->
  <img width="1075" alt="Multimodal Routing Architecture"
       src="figures/model_architecture.png" />
</p>

---

## Table of Contents

1. [Overview](#overview)  
2. [Key Contributions](#key-contributions)  
3. [Clinical Prediction Tasks](#clinical-prediction-tasks)  
4. [Method Overview](#method-overview)  
5. [Missing-Modality Robustness and Auditing](#missing-modality-robustness-and-auditing)  
6. [Data Sources](#data-sources-credentialed-access-required)  
7. [Repository Structure](#repository-structure)  
8. [Citation](#citation)  
9. [Disclaimer](#disclaimer)

---

## Overview

Electronic Health Records (EHRs) are inherently **multimodal**, combining heterogeneous sources such as structured longitudinal measurements, free-text clinical notes, and medical images. While multimodal models often improve predictive performance, most existing approaches rely on **deep fusion**, obscuring how individual modalities and their interactions contribute to predictions.

**Multimodal Routing** is a routing-based multimodal learning framework that makes multimodal reasoning **explicit, interpretable, and auditable**. The model separates unimodal signals from directional cross-modal interactions and learns **how**, **when**, and **for which outcomes** each evidence pathway contributes.

The framework operates over three core clinical modalities:

- **L** — Structured longitudinal data (vitals, laboratory values, medications)  
- **N** — Clinical notes  
- **I** — Chest X-ray images  

By explicitly modeling multimodal routes and dynamically weighting them at inference time, the framework supports:

- transparent multimodal reasoning  
- robustness to missing modalities  
- auditing of model decision pathways in realistic clinical settings  

---

## Key Contributions

- **Explicit modeling of 10 multimodal routes**
  - **3 unimodal routes**
  - **6 directional bimodal routes**  
    (e.g., `N ← L` vs. `L ← N`)
  - **1 trimodal route**

- **Route activations (patient-specific)**  
  Quantify how strongly each unimodal or cross-modal route is expressed for an individual patient.

- **Routing coefficients (patient- and label-specific)**  
  Quantify how much each route contributes to each prediction target.

- **Inference-time route masking**  
  Simulates missing-modality scenarios without retraining by disabling routes involving unavailable modalities and renormalizing routing weights.

- **Auditable multimodal reasoning**  
  Routing weight redistribution under missing-modality settings enables systematic auditing of modality reliance, robustness, and potential shortcut learning.

---

## Clinical Prediction Tasks

This codebase supports two ICU prediction tasks using paired **tri-modal** EHR data.

### Binary ICU Mortality Prediction
- Observation window: **first 48 hours** of the ICU stay

### Multi-label Phenotype Prediction
- **25 phenotypes**
- Observation window: **full ICU stay**
- **Discharge summaries are excluded** to avoid information leakage

---

## Method Overview

### Multimodal Routes

The model explicitly constructs a set of **interpretable multimodal routes**.

#### Unimodal routes
{ L, N, I }

#### Directional bimodal routes
{ L ← N, N ← L, L ← I, I ← L, N ← I, I ← N }

#### Trimodal route
{ LNI }


The trimodal route is constructed **hierarchically from paired directional interactions**.

Each route \( r \in \mathcal{R} \) produces a route-specific embedding \( e_r \) representing a distinct unimodal or cross-modal evidence pathway.

---

### Decision Mechanism

For patient <b>b</b> and prediction target (label) <b>c</b>, the decision representation is:

<p align="center">
  <b>d</b><sub>b,c</sub> = &sum;<sub>r &isin; &#8475;</sub> R<sub>b,r,c</sub> &middot; &alpha;<sub>b,r</sub> &middot; <b>&#7869;</b><sub>r</sub>
</p>

where:

- <b>&alpha;</b><sub>b,r</sub> — route activation (patient-specific)  
- <b>R</b><sub>b,r,c</sub> — routing coefficient (patient- and label-specific), with  
  <p align="center">&sum;<sub>r &isin; &#8475;</sub> R<sub>b,r,c</sub> = 1</p>
- <b>&#7869;</b><sub>r</sub> — primary route representation (content vector)

The effective route contribution is:

<p align="center">
  <b>W</b><sub>b,r,c</sub> = &alpha;<sub>b,r</sub> &middot; R<sub>b,r,c</sub>
</p>


The **effective route contribution** is defined as:

<p align="center">
  <b>W</b><sub>b,r,c</sub> = <b>α</b><sub>b,r</sub> · <b>R</b><sub>b,r,c</sub>
</p>

This formulation enforces **structured, selective, and interpretable multimodal aggregation**, ensuring predictions are driven only by routes that are both strongly expressed for a patient and relevant to the target outcome.

---

## Missing-Modality Robustness and Auditing

Robustness is evaluated using **inference-time route masking**.

When a modality is missing:
- all routes involving that modality are disabled  
- remaining routing coefficients are renormalized  
- inference proceeds **without retraining or imputation**

Changes in routing weights and predictive performance under missing-modality scenarios provide an **audit signal** that reveals:

- modality essentiality  
- reliance on narrow vs. complementary evidence  
- potential shortcut learning  

---

## Data Sources (Credentialed Access Required)

This project uses the following PhysioNet datasets:

- **MIMIC-IV**
- **MIMIC-IV-Note**
- **MIMIC-CXR-JPG**
---

## Repository Structure

The repository is organized by **prediction task**, with parallel pipelines for ICU mortality and phenotype prediction.

```text
MultimodalRouting/
├── Data/                         # (Local only) processed data placeholders
│
├── INSPECT/                      # Inspection / debugging utilities (experimental; not used in final results)
│
├── MIMIC-IV/
│   ├── Data/                     # Task-specific data loading, pairing, and preprocessing
│   ├── Model/                    # Shared utilities (training loops, losses, evaluation, helpers)
│   │
│   ├── MortModel/                # Binary ICU mortality prediction (first 48 hours)
│   │   ├── Baseline/             # Unimodal and standard fusion baselines
│   │   │
│   │   ├── Paired_Cross_Attention/
│   │   │   ├── encoders.py       # Modality-specific encoders (L, N, I)
│   │   │   ├── mult_model.py     # Multimodal routing backbone with explicit route construction
│   │   │   ├── routing_and_heads.py
│   │   │   │   # Route activations (α) and routing coefficients (R)
│   │   │   │   # Binary mortality prediction head
│   │   │   ├── capsule_layers.py # Capsule-style route aggregation and normalization
│   │   │   ├── transformer.py    # Cross-attention modules for directional routes
│   │   │   ├── env_config.py     # Experiment configuration and hyperparameters
│   │   │   └── main.py           # Training / evaluation entry point
│   │   │
│   │   ├── Paired_Simple_Concat/ # Undirected fusion ablation (no routing, no directionality)
│   │   └── Partial/              # Inference-time missing-modality and route-masking experiments
│   │
│   └── PhenoModel/               # Multi-label phenotype prediction (full ICU stay)
│       ├── Baseline/             # Unimodal and late/joint fusion baselines
│       │
│       ├── Paired_Cross_Attention/
│       │   ├── encoders.py       # Modality-specific encoders (L, N, I)
│       │   ├── mult_model.py     # Multimodal routing backbone (same routes as mortality)
│       │   ├── routing_and_heads.py
│       │   │   # Route activations (α) and routing coefficients (R)
│       │   │   # Multi-label phenotype prediction heads (25 phenotypes)
│       │   ├── capsule_layers.py # Capsule routing and aggregation logic
│       │   ├── multhead_attention.py
│       │   │   # Directional cross-attention for paired routes
│       │   ├── position_embedding.py
│       │   │   # Positional encoding for longitudinal and text sequences
│       │   ├── transformer.py    # Transformer blocks for cross-modal interactions
│       │   ├── env_config.py     # Phenotype-specific configuration and hyperparameters
│       │   └── main.py           # Training/evaluation entry point
│       │
│       ├── Paired_Simple_Concat/ # Undirected bimodal fusion ablation
│       └── Partial/              # Missing-modality robustness and auditing experiments
│
├── figures/
│   ├── model_architecture.png    # Architecture figure (rendered in README)
│
└── README.md
---

## Core Routing Implementations (Clickable)

The following links provide direct navigation to the **core multimodal routing implementations**
used for ICU mortality and phenotype prediction.

---

### Mortality · Paired_Cross_Attention  
**Binary ICU mortality prediction (first 48 hours)**  

📁 **Folder:**  
[`MIMIC-IV/MortModel/Paired_Cross_Attention/`](MIMIC-IV/MortModel/Paired_Cross_Attention/)

**Key files:**
- [`main.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/main.py) — training & evaluation entry point  
- [`env_config.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/env_config.py) — experiment configuration & hyperparameters  
- [`encoders.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/encoders.py) — modality-specific encoders (L, N, I)  
- [`mult_model.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/mult_model.py) — multimodal routing backbone and route construction  
- [`routing_and_heads.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/routing_and_heads.py) — route activations (α), routing coefficients (R), mortality head  
- [`capsule_layers.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/capsule_layers.py) — capsule-style route aggregation and normalization  
- [`transformer.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/transformer.py) — directional cross-attention modules  

---

### Phenotype · Paired_Cross_Attention  
**Multi-label phenotype prediction (25 phenotypes, full ICU stay)**  

📁 **Folder:**  
[`MIMIC-IV/PhenoModel/Paired_Cross_Attention/`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/)

**Key files:**
- [`main.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/main.py) — training & evaluation entry point  
- [`env_config.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/env_config.py) — phenotype-specific configuration  
- [`encoders.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/encoders.py) — modality-specific encoders (L, N, I)  
- [`mult_model.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/mult_model.py) — multimodal routing backbone (shared route logic)  
- [`routing_and_heads.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/routing_and_heads.py) — route activations (α), routing coefficients (R), phenotype heads  
- [`capsule_layers.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/capsule_layers.py) — capsule routing and aggregation  
- [`multhead_attention.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/multhead_attention.py) — directional multi-head cross-attention  
- [`position_embedding.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/position_embedding.py) — temporal & positional embeddings  
- [`transformer.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/transformer.py) — transformer blocks for cross-modal interactions  

---

### Design Notes

- **MortModel** and **PhenoModel** share the same routing architecture but differ in:
  - prediction head (binary vs. multi-label)
  - observation window (48h vs. full ICU stay)
- **Paired_Cross_Attention** implements the full multimodal routing framework:
  - unimodal, directional bimodal, and trimodal routes  
  - route activations (α) and routing coefficients (R)
- **Paired_Simple_Concat** removes directionality and routing (ablation)
- **Partial** evaluates inference-time missing-modality robustness via route masking

