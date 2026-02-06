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
