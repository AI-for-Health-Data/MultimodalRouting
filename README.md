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
