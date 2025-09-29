# Preprocessing pipeline

## Introduction

This repository provides the data preprocessing pipeline used in the **MedFuse** model, covering **EHR mortality**, **EHR phenotyping**, and **CXR selection**.  
The preprocessing code performs two types of splits:

- **Pair split**: Contains patients with both EHR and CXR data.  
- **Partial split**: Contains EHR data without corresponding CXR data together with pair split contents.  

For **validation**, while the pipeline also generates a partial split, only the **paired data** are included in the validation process to ensure consistent evaluation.


### Related Data

The preprocessing combines **EHR** and **CXR** information using:

- `mimic-cxr-2.0.0-metadata.csv`  
- `all_stays.csv`

The file **`all_stays.csv`** is derived from the EHR tables:

- `patients.csv`  
- `admissions.csv`  
- `icustays.csv`  

with the following processing steps:

- Remove ICU stays with **transfers**  
- Remove subjects with **multiple ICU stays**  
- Compute and record both **in-unit** and **in-hospital** mortality labels  
- Apply **age filtering**: keep only patients aged **≥18**  

The preprocessing script that generates `all_stays.csv` can be found here:  
[MedFuse mimic4extract code](https://github.com/nyuad-cai/MedFuse/tree/main/mimic4extract)


### Phenotyping (CXR-based)
- **Matching window:** `intime <= StudyDateTime <= outtime`  
- **Split statistics (paired data):**  
  - Train = 8,255  
  - Validation = 859  
  - Test = 2,116  
  - Total = 11,230  
- **Split statistics (partial data, EHR only):**  
  - Train = 60,909  
  - Validation = 4,756  
  - Test = 11,845  
  - Total = 77,510  


### In-Hospital Mortality (48h window)
- **Matching window:** `intime <= StudyDateTime <= intime + 48h`  
- **Split statistics (paired data):**  
  - Train = 4,956  
  - Validation = 490  
  - Test = 1,231  
  - Total = 6,677  
- **Split statistics (partial data, EHR only):**  
  - Train = 29,171  
  - Validation = 2,161  
  - Test = 5,302  
  - Total = 36,634  
  
---  

## Guide

### Usage

**datasets/fusion.manifest_partial_ehr_cxr** generates CSV manifest files that describe the final splits:

The split type is controlled by the `--data_pairs` argument:

- `--data_pairs partial_ehr_cxr` → generates partial EHR-only split  
- `--data_pairs paired_ehr_cxr` → generates paired EHR–CXR split

- `--task in-hospital-mortality --labels_set mortality` → generates mortality result, else phenotyping result


### Output

- **`cxr_debug_pheno/mortality.csv`**  
  Contains valid **paired** cxr data.  
- **`partial_ehr_cxr_all_pheno.csv`**  
  Contains valid **partial** cxr data.


- **`train/val/test_metadata_partial_pheno/mortality.csv`**  
  Contains the metadata for the **partial** split.  
- **`train/val/test_metadata_pheno/mortality.csv`**  
  Contains the metadata for the **paired** split.

---

## Source

[MedFuse mimic4extract code](https://github.com/nyuad-cai/MedFuse)