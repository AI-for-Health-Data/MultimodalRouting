## MedFuse-Aligned Cohort: MIMIC-IV (EHR + CXR + Notes)

- 17 structured time-series variables binned over the first 48h of ICU stay
- Optional CXR pairing (last in stay for phenotyping; last in 0–48h for IHM)
- Optional notes aggregation (0–48h), chunked to BioClinicalBERT 512-token inputs
- Patient-level train/val/test splits or exact MedFuse splits (if listfiles provided)
- Train-only normalization stats
  
- **Population filters**
  - Adults only (≥18 years)
  - First ICU stay per patient only
  - ICU stay must last at least **48 hours**

- **Time windowing**
  - First **48 hours** of ICU stay
  - Aggregated into **24 bins** of 2 hours each
  - Each bin contains up to 17 structured variables (labs + vitals)
  - Missing values forward-filled within each stay

- **Variables (17 total)**
  - *Vitals*: HeartRate, SysBP, DiasBP, MeanBP, RespRate, Temperature, SpO₂  
  - *Labs*: Sodium, Potassium, Chloride, Bicarbonate, BUN, Creatinine, Glucose, Hematocrit, WBC, Platelets  
  - Units normalized 

- **Labels**
  - **Phenotyping (multi-label CCS)**  
    - CCS categories from ICD-9/10 codes  
    - Optional COPD/bronchiectasis binary label
  - **In-Hospital Mortality (IHM)**  
    - Stays with death **within first 48h** dropped (to avoid label leakage)  
    - Label = death **after 48h but before discharge**

- **CXR pairing**
  - **Phenotyping** → last CXR in the stay  
  - **IHM** → last CXR taken in the first 48h  
  - `--ap_only` flag restricts to AP views  
  - Image file paths resolved against `mimic-cxr-jpg/2.0.0/files/...`

- **Notes**
  - Notes from *radiology.csv.gz* and *discharge.csv.gz*  
  - Restricted to 0–48h window after ICU admission  
  - Concatenated per stay, cleaned, and chunked into **BioClinicalBERT-ready 512-token segments**

- **Splits**
  - If MedFuse listfiles provided → exact reproduction of MedFuse train/val/test splits  
  - Else → random **patient-level** split (70/10/20)

- **Normalization**
  - Train-only mean/median/std statistics  
  - Structured features z-scored  
  - Optional categorical → one-hot

---

## The Three Notebooks
[`build_cohort.py`](MIMIC-IV/cohort/build_cohort.py)
### 1. |[`build_cohort.py`](MIMIC-IV/cohort/build_cohort.py)|: 
filters, labels, splits, CE/LE extraction, 48h binning, CXR pairing, notes chunking, and normalization.
Required inputs (all in one folder):
'admissions.csv.gz, patients.csv.gz, diagnoses_icd.csv.gz,
icustays.csv.gz, chartevents.csv.gz, labevents.csv.gz,
d_items.csv.gz, d_labitems.csv.gz, varmap_mimiciv_17.csv' 
- mimic-cxr-2.0.0-metadata.csv.gz, --cxr_files_root (jpg files root), --ap_only
- Provide --split_listfiles_dir containing train_listfile.csv, val_listfile.csv, test_listfile.csv
- Runs the cohort builder (`build_cohort.py`)
- Handles filters, labels, splits, and CXR pairing

### 2. `build_cohort.py`
- Extracts 17 variable mappings from `d_items.csv.gz` and `d_labitems.csv.gz`
- Resolves multiple itemids with priority order
- Defines canonical target units for normalization

### 3. `export_model_inputs.ipynb`
- Converts NPZ + master into **model-ready parquets**:
  - `structured_24h.parquet` — 24×F per stay (z-scored/one-hot expanded)
  - `images_24h.parquet` — stay_id, image_path (paired only)
  - `notes_24h.parquet` — stay_id, text (512-token chunks)
  - `labels.parquet` — mort, plus placeholders pe, ph
  - `splits.json` — lists of stay_id for train/val/test


