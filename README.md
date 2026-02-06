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
## Architecture

*See the architecture figure above.*

The **Multimodal Routing** framework explicitly decomposes multimodal reasoning into
**unimodal**, **directional bimodal**, and **trimodal** evidence pathways, enabling
transparent, robust, and auditable clinical prediction.

The implementation is shared across tasks, with task-specific prediction heads for
mortality and phenotype prediction.

---

### Modality Encoders

Each clinical modality is encoded independently before any fusion:

- **L — Structured longitudinal data**  
  Time-ordered vitals, laboratory values, and medications are encoded using a
  transformer-based sequence model.

  - Encoder implementation:  
    - Mortality:  
      [`MIMIC-IV/MortModel/Paired_Cross_Attention/encoders.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/encoders.py)  
    - Phenotypes:  
      [`MIMIC-IV/PhenoModel/Paired_Cross_Attention/encoders.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/encoders.py)

- **N — Clinical notes**  
  Free-text notes are encoded using **BioClinicalBERT**, followed by chunk-level pooling.

  - Encoder implementation:  
    - Mortality:  
      [`MIMIC-IV/MortModel/Paired_Cross_Attention/encoders.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/encoders.py)  
    - Phenotypes:  
      [`MIMIC-IV/PhenoModel/Paired_Cross_Attention/encoders.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/encoders.py)

- **I — Chest X-ray images**  
  Radiographs are encoded using a convolutional backbone to produce spatial feature tokens.

  - Encoder implementation:  
    - Mortality:  
      [`MIMIC-IV/MortModel/Paired_Cross_Attention/encoders.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/encoders.py)  
    - Phenotypes:  
      [`MIMIC-IV/PhenoModel/Paired_Cross_Attention/encoders.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/encoders.py)

Encoders are trained jointly, but **no implicit fusion** occurs at this stage.

---

### Multimodal Route Construction

From the unimodal embeddings, the model constructs **10 explicit routes**:

- **Unimodal routes:**  
  `L`, `N`, `I`

- **Directional bimodal routes:**  
  `L ← N`, `N ← L`,  
  `L ← I`, `I ← L`,  
  `N ← I`, `I ← N`

- **Trimodal route:**  
  `LNI`, constructed hierarchically from paired directional interactions

Directional bimodal routes are implemented using **cross-attention**, where one modality
acts as the query and another as the key/value, preserving **directionality and asymmetry**
in cross-modal influence.

- Cross-attention backbone:
  - Mortality:  
    [`MIMIC-IV/MortModel/Paired_Cross_Attention/mult_model.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/mult_model.py)
  - Phenotypes:  
    [`MIMIC-IV/PhenoModel/Paired_Cross_Attention/mult_model.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/mult_model.py)

- Directional attention modules:
  - Mortality:  
    [`MIMIC-IV/MortModel/Paired_Cross_Attention/transformer.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/transformer.py)
  - Phenotypes:  
    [`MIMIC-IV/PhenoModel/Paired_Cross_Attention/multhead_attention.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/multhead_attention.py)

Each route produces a **route-specific embedding** representing a distinct unimodal or
cross-modal evidence pathway.

---

### Routing, Activations, and Aggregation

Each route embedding is passed through a capsule-style routing mechanism that learns:

- **Route activations (α):**  
  Patient-specific scalars indicating how strongly each route is expressed.

- **Routing coefficients (R):**  
  Patient- and label-specific weights determining how much each route contributes
  to a given prediction target.

The routing logic and prediction heads are implemented in:

- Mortality:
  - [`MIMIC-IV/MortModel/Paired_Cross_Attention/routing_and_heads.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/routing_and_heads.py)
- Phenotypes:
  - [`MIMIC-IV/PhenoModel/Paired_Cross_Attention/routing_and_heads.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/routing_and_heads.py)

Capsule-style normalization and aggregation:
- [`capsule_layers.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/capsule_layers.py)

Only routes that are **both strongly expressed for a patient** and **relevant to the
target label** meaningfully contribute to the final prediction.

---

### Task-Specific Prediction Heads

The routing backbone is shared across tasks, with task-specific heads:

- **Binary ICU mortality prediction** (first 48 hours):
  - Entry point:  
    [`MIMIC-IV/MortModel/Paired_Cross_Attention/main.py`](MIMIC-IV/MortModel/Paired_Cross_Attention/main.py)

- **Multi-label phenotype prediction** (25 phenotypes, full ICU stay):
  - Entry point:  
    [`MIMIC-IV/PhenoModel/Paired_Cross_Attention/main.py`](MIMIC-IV/PhenoModel/Paired_Cross_Attention/main.py)

---

### Architectural Properties

- **Interpretability:**  
  Predictions decompose into explicit unimodal and cross-modal route contributions.

- **Robustness:**  
  Missing modalities are handled at inference time by disabling affected routes
  (see [`Partial/`](MIMIC-IV/MortModel/Partial/)).

- **Auditability:**  
  Routing weight redistribution under missing-modality scenarios reveals modality
  reliance, redundancy, and potential shortcut learning.

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
