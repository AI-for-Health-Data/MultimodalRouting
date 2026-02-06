<p align="center">
  <img src="figures/model_architecture.png" width="900" />
</p>
---

## Overview

This repository contains the official implementation of **Multimodal Routing**, a routing-based multimodal learning framework for clinical prediction from Electronic Health Records (EHRs).

Electronic Health Records are inherently **multimodal**, combining heterogeneous sources such as structured longitudinal measurements, free-text clinical notes, and medical images. While multimodal models often improve predictive performance, most existing approaches rely on deep fusion strategies that obscure how individual modalities contribute to predictions and limit interpretability.

**Multimodal Routing** addresses this limitation by explicitly modeling **unimodal**, **directional bimodal**, and **trimodal** evidence pathways across three core clinical modalities:

- **L** — Structured longitudinal data (vitals, laboratory measurements, medications)
- **N** — Clinical notes
- **I** — Chest X-ray images

By learning *how* and *when* each modality—and their interactions—contributes to predictions, the framework enables:

- **Interpretable multimodal reasoning**
- **Robustness to missing modalities at inference time**
- **Auditable clinical decision-making**

---

## Associated Paper

**Multimodal Routing for Interpretable, Robust, and Auditable Clinical Prediction**  
ACM Conference on Connected Health (CHASE), under review

If you use this code, please cite the associated paper.
