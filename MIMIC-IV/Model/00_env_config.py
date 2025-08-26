# This notebook sets up the environment and shared configuration for the **MultimodalRouting** project.
## **Modalities:** L (structured labs/vitals), N (notes), I (images)  
## **Tasks:** mortality, pulmonary embolism (PE), pulmonary hypertension (PH)

## It also defines the route/block taxonomy and a few helper utilities used across other notebooks.

## **Routing backends:**
### - `"ema_logits"` — 7 route **logit** heads + task-wise EMA loss→weight routing (two-stage: route→block).
### - `"capsule_features"` — 7 route **feature** heads + **capsule** dynamic routing from route features to task concepts (per-sample couplings).
