# Multimodal Routing for Interpretable, Robust, and Auditable Clinical Prediction

This repository contains the official implementation of **Multimodal Routing**, a routing-based multimodal learning framework for clinical prediction from Electronic Health Records (EHRs).

The model explicitly separates and weighs **unimodal**, **directional bimodal**, and **trimodal** evidence pathways across three clinical modalities:

- **L** — Structured longitudinal data (vitals, labs, medications)
- **N** — Clinical notes
- **I** — Chest X-ray images

By learning *how* and *when* each modality (and their interactions) contributes to predictions, the framework enables **interpretability**, **robustness to missing data**, and **auditable multimodal reasoning**, which are critical for deployment in high-stakes clinical settings.

---

## Associated Paper

**Multimodal Routing for Interpretable, Robust, and Auditable Clinical Prediction**  

If you use this code, please cite the associated paper.
## Key Contributions

- **Explicit modeling of 10 multimodal routes**
  - **3 unimodal** routes
  - **6 directional bimodal** routes  
    (e.g., \(N \leftarrow L\) vs. \(L \leftarrow N\))
  - **1 trimodal** route

- **Route activations (patient-specific)**  
  Quantify how strongly each unimodal or cross-modal route is expressed for an individual patient.

- **Routing coefficients (patient- and label-specific)**  
  Quantify how much each route contributes to each prediction target.

- **Inference-time route masking**  
  Simulates missing-modality scenarios without retraining by disabling routes involving unavailable modalities and renormalizing routing weights.

- **Auditable multimodal reasoning**  
  Routing weight redistribution under missing-modality settings enables systematic auditing of modality reliance and model robustness.

---

## Clinical Prediction Tasks

This code supports two ICU prediction tasks using paired **tri-modal** EHR data:

### Binary ICU Mortality Prediction
- Observation window: **first 48 hours** of the ICU stay

### Multi-label Phenotype Prediction
- **25 phenotypes**
- Observation window: **full ICU stay**
- **Discharge summaries are excluded** to avoid information leakage

---
### 2. **Repository tree must be inside a code block**
Right now it is rendered as plain text, which will break alignment on GitHub.

🔧 **Fix**: wrap the repository structure in triple backticks.

```markdown
MultimodalRouting/
├── Data/
├── INSPECT/
├── MIMIC-IV/
│ ├── Data/
│ ├── Model/
│ ├── MortModel/
│ │ ├── Baseline/
│ │ ├── Paired_Cross_Attention/
│ │ ├── Paired_Simple_Concat/
│ │ └── Partial/
│ └── PhenoModel/
│ ├── Baseline/
│ ├── Paired_Cross_Attention/
│ ├── Paired_Simple_Concat/
│ ├── Partial/
│ └── README.md
└── README.md

### Design Notes

- **MortModel** and **PhenoModel** follow the same architectural logic but differ in:
  - prediction head (binary vs. multi-label)
  - observation window (48h vs. full stay)
- **Paired_Cross_Attention** implements the full routing framework:
  - unimodal, directional bimodal, and trimodal routes
  - routing activations and routing coefficients
- **Paired_Simple_Concat** removes directional cross-attention and serves as an ablation
- **Partial** contains inference-time route masking experiments for missing-modality auditing
- Structured data, notes, and images are always processed by **separate encoders** and fused only through explicit routing

---
## Method Overview

### Model Architecture

The figure below illustrates the overall architecture of the proposed **multimodal routing framework**, including modality-specific encoders, explicit unimodal and cross-modal routes, routing activations, and label-specific routing coefficients.

<p align="center">
  <img src="figures/model_architecture.png" width="900">
</p>

<p align="center">
  <b>Figure 1:</b> Architecture of the multimodal routing framework. Structured longitudinal data (L),
  clinical notes (N), and chest X-rays (I) are encoded separately. The model constructs unimodal,
  directional bimodal, and trimodal routes using cross-attention. Route activations (patient-specific)
  and routing coefficients (patient- and label-specific) selectively aggregate route representations
  to produce interpretable and robust predictions.
</p>

---

### Multimodal Routes

The model explicitly constructs a set of **interpretable multimodal routes** to represent unimodal signals and cross-modal interactions:

- **Unimodal routes**  
  \[
  \{L,\; N,\; I\}
  \]

- **Directional bimodal routes**  
  \[
  \{L \leftarrow N,\; N \leftarrow L,\; L \leftarrow I,\; I \leftarrow L,\; N \leftarrow I,\; I \leftarrow N\}
  \]

- **Trimodal route**  
  \[
  \{LNI\}
  \]
  constructed hierarchically from paired directional interactions.

Each route \(r \in \mathcal{R}\) produces a route-specific embedding \(e_r\) corresponding to a distinct unimodal or cross-modal evidence pathway.

---

### Decision Mechanism

For a patient \(b\) and prediction target (label) \(c\), the final decision representation is computed as:

\[
\mathbf{d}_{b,c}
=
\sum_{r \in \mathcal{R}}
R_{b,r,c} \cdot \alpha_{b,r} \cdot \tilde{\mathbf{e}}_r
\]

where:

- **\(\alpha_{b,r}\)** — *Route activation*  
  Patient-specific scalar indicating how strongly route \(r\) is expressed.

- **\(R_{b,r,c}\)** — *Routing coefficient*  
  Patient- and label-specific weight indicating how much route \(r\) contributes to label \(c\),
  normalized across routes.

- **\(\tilde{\mathbf{e}}_r\)** — *Primary route representation*  
  Content vector carrying predictive information from route \(r\).

The **effective route contribution** is defined as:
\[
W_{b,r,c} = \alpha_{b,r} \cdot R_{b,r,c}
\]

This formulation enforces **structured, selective, and interpretable multimodal aggregation**, ensuring
that predictions are driven only by routes that are both strongly expressed for a patient and relevant
to the target outcome.
