# Multimodal Routing on MIMIC‑IV

This repository implements a **multimodal clinical predictor** trained on MIMIC‑IV with three input modalities:

* **L** — structured EHR time‑series (labs + vitals)
* **N** — clinical notes
* **I** — chest X‑rays

The system builds **7 routes** — `L, N, I, LN, LI, NI, LNI` — and a **router** that adaptively assigns per‑patient weights to each route. A final concat head consumes the **gated route embeddings** to output task‑specific predictions.

Supported labels:

* `mort` — in‑hospital mortality
* `pe` — pulmonary embolism
* `ph` — pulmonary hypertension

---

## Repository layout

```
MIMIC-IV/Model/
├── env_config.py                # Global config, environment + hyperparameters
├── encoders.py                  # Structured, notes, and image encoders; MulT-style fusions
├── routing_and_heads.py         # Fusion modules, route heads, router, final concat head
├── train_step1_unimodal.py      # Step 1: unimodal encoder + head training
├── train_step2_bimodal.py       # Step 2: bimodal fusion + head training
├── train_step3_trimodal_router.py # Step 3: trimodal + router + final concat head training
├── evaluation_metrics.py        # Test‑time evaluation (AUROC, AP, gate stats, fairness)
├── inference_demo.py            # Load checkpoints and run batch inference demo
└── interpretability.py          # Route contributions, UC/BI/TI decomposition, dataset‑wide analysis
```

---

## Data expectations

Produced by cohort builder and stored under `./data/MIMIC-IV`:

* `structured_24h.parquet` — 24 time bins × variables per ICU stay
* `notes_24h.parquet` — clinical notes in first 24h
* `images_24h.parquet` — CXR paths per stay
* `labels.parquet` — label columns (`mort`, `pe`, `ph`)
* `splits.json` — patient‑level train/val/test splits

---

## Training pipeline

Training proceeds in **3 steps**:

1. **Unimodal (Step 1)**: train encoders + heads for `L`, `N`, `I`
2. **Bimodal (Step 2)**: freeze encoders; train fusions + heads for `LN`, `LI`, `NI`
3. **Trimodal + Router (Step 3)**: build all 7 routes, compute gates, train final concat head

---

## Inference demo

`inference_demo.py` runs a quick demo on the **test split**:

* Loads checkpoints from all 3 steps
* Builds encoders, fusions, route heads, router, and final head
* Computes **route embeddings**, **masks**, **gates**, and **final prediction**
* Prints predicted probabilities for first few patients and detailed gates for one example

Outputs include:

* Predicted probabilities for first *k* patients
* Route gates and block‑level means for a selected sample

---

## Interpretability

`interpretability.py` provides analysis utilities:

* **Route contributions (occlusion):** compute logit change if a route is removed
* **Block weights:** aggregate gates into uni/bi/tri‑modal block weights
* **UC/BI/TI decomposition:** unimodal, bimodal, trimodal interaction estimates using expectation baselines
* **Dataset summaries:** compute global averages of contributions, UC/BI/TI decomposition, and mean embeddings

Outputs:

* Per‑sample DataFrame with gates, contributions, embedding norms
* Global summary of |Δlogit| per route
* UC/BI/TI decomposition with dataset‑wide means

---

## How multimodal routing works

1. **Route embeddings** are built for 7 modality subsets
2. **Route heads** map each embedding to a logit
3. **Router** computes gates:

   * `uniform` → equal over available routes
   * `learned` → MLP on unimodal embeddings
   * `loss_based` → weight inversely to BCE loss per route
4. **Final concat head** consumes gated route embeddings → final logit

---

## Configuration

All hyperparameters and paths are in `env_config.py`. Key fields:

* `d` — embedding dim (default 256)
* `structured_seq_len`, `structured_n_feats`
* `text_model_name`, `max_text_len`
* `image_model_name`
* `gate_mode`, `loss_gate_alpha`, `l2norm_each`
* `data_root`, `ckpt_root`, `task_name`

---

## Interpretability outputs

* **Route gates per sample** → which modalities dominate decisions
* **Occlusion contributions** → per‑route impact on final logit
* **UC/BI/TI decomposition** → disentangle unimodal, bimodal, trimodal interactions

---

## `inference_demo.py`

**Purpose.** Load the 3 checkpoints for a given task, run a single test batch, print predicted probabilities, and **inspect the per‑sample route gates** (and block‑level gate means: uni/bi/tri).

### it builds

* **Stack**: `build_stack()` constructs encoders (L/N/I), pairwise/trimodal fusion MLPs, all 7 `RouteHead`s, a `RouteGateNet` (learned router), and the `FinalConcatHead`.
* **Checkpoints**: `load_checkpoints()` loads

  * Step‑1 encoders + L/N/I heads, Step‑2 fusions + LN/LI/NI heads
  * Step‑3: retrieves `gate_mode`, `loss_gate_alpha`, `l2norm_each`, loads `gate_net`, final head, and optional `LNI` fusion/head
* **Masks**: `build_masks()` flags which modalities are present; gates are normalized over **available** routes only.

### Routing in the demo

* `_compute_gates(...)` supports the three modes exactly like training:

  * `uniform`: average over available routes
  * `learned`: MLP router on `[zL|zN|zI]` with masking
  * `loss_based`: per‑route BCE (using frozen heads) → softmax of `-α·loss` with masking

### End‑to‑end forward

1. Encode unimodal `zL/zN/zI` for the first batch
2. Build the 7 route embeddings via `make_route_inputs(z, fusion)`
3. Compute `gates` with `_compute_gates(...)`
4. Concatenate **gated** routes with `concat_routes(...)` → `[B, 7·d]`
5. Predict with `FinalConcatHead` and print top‑K probabilities and gate breakdown for one sample

---

## `interpretability.py`

**Purpose.** Provide a small toolkit to analyze **route contributions** and a **UC/BI/TI** (Unimodal Contrast / Bimodal Interaction / Trimodal Interaction) decomposition using dataset‑mean embeddings as baselines.

### Key APIs

* `embeddings_from_batch(...)` → `{"L": zL, "N": zN, "I": zI}` pooled embeddings
* `forward_full(...)` → end‑to‑end: build routes → compute gates → concat → final logits; returns `(ylogits, gates, routes_raw, routes_weighted)`
* `route_contributions_occlusion(...)`:

  * Occlusion‑style importance: set each route’s gate to 0 (one at a time), recompute, and take `Δ = y_full − y_wo_route`
  * Returns the full prediction and a dict of **per‑route logit deltas**
* `block_weights_from_gates(gates)` → compress 7‑way gates into 3 blocks: `uni`, `bi`, `tri`
* `compute_dataset_means(loader, ...)` → estimates μL/μN/μI from the dataset (used as substitution baselines)
* `uc_bi_ti_for_batch(...)` and `collect_uc_bi_ti(...)` → UC/BI/TI decomposition at sample and dataset levels
* `collect_contributions(...)` → builds a tidy DataFrame with per‑sample fields:

  * `y_true, y_prob, y_logit`
  * `block_w__{uni,bi,tri}`
  * For each route r∈{L,N,I,LN,LI,NI,LNI}: `gate__r`, `route_contrib__r`, `route_emb_norm__r`
* `global_mean_abs_contrib(df)` → scalar summary: mean |Δ(logit)| per route

### UC/BI/TI intuition

Let `F(zL,zN,zI)` be the final **logit** function. With dataset means `(μL,μN,μI)` as neutral baselines, we form:

* **Unimodal contrasts** (replace other modalities by means):

  * `F_Lmm, F_mNm, F_mmI`
* **Pairwise contrasts**: `F_LNm, F_LmI, F_mNI`
* **Full**: `F_full = F(zL, zN, zI)` and `F_mmm = F(μL, μN, μI)`
* Decomposition:

  * `UC = F_Lmm + F_mNm + F_mmI − 2·F_mmm`
  * `BI = (F_LNm − F_Lmm − F_mNm + F_mmm) + (F_LmI − F_Lmm − F_mmI + F_mmm) + (F_mNI − F_mNm − F_mmI + F_mmm)`
  * `TI = F_full − UC − BI`

This separates **main effects** (per modality), **pairwise interactions**, and **trimodal synergy** at the **logit level**.

---

## Local vs. Global Interpretability

If you run the interpretability notebook, you can get both levels of explanation:

* **Local (per‑patient):** per‑sample gates, occlusion contributions (Δlogit when removing a route), and UC/BI/TI decomposition. These explain why a single patient’s prediction looks the way it does.
* **Global (dataset‑level):** average gates, global mean |Δlogit| per route, and mean UC/BI/TI values across the test set. These explain what the model relies on overall.

---

## End‑to‑end usage recap

1. Train: Step‑1 → Step‑2 → Step‑3 for your task (`mort|pe|ph`)
2. Evaluate: `evaluation_metrics.py` to validate AUROC/AP and see mean gate usage
3. Inspect: `inference_demo.py` to print probabilities and **per‑sample gates**
4. Explain: `interpretability.py` to compute **route contributions** and the **UC/BI/TI** decomposition

This rounds out the full multimodal routing workflow: **build 7 routes**, **learn gates**, **predict**, and **interpret** which routes carried the signal for each patient and globally.
