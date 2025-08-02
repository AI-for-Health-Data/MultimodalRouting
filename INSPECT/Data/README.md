# INSPECT Dataset: Cohort Construction, EHR Filtering, and Label Merging

---

## 1. Cohort Construction

**Script:** `Data/00_build_INSPECT_full_cohort_with_impressions.py`

### Steps:

* Load imaging metadata, person IDs, and procedure times.
* Merge labels and train/valid/test splits.
* Join in the radiology impression text from `Final_Impressions.csv`.
* Filter EHR tables to events *before or on* the imaging study time.

**Output Files:**

* `INSPECT_public/cohort/inspect_cohort.csv`
* `INSPECT_public/reports/inspect_impressions.csv`
* `INSPECT_full_dataset_with_impressions.csv.gz`
* Filtered EHR tables in `filtered_ehr/`

### Table 4: Label Counts

| Task            | Label    | Count  |
| --------------- | -------- | ------ |
| PE (NLP)        | True     | 4,684  |
|                 | False    | 18,552 |
| 1M Mortality    | True     | 1,198  |
|                 | False    | 20,793 |
|                 | Censored | 1,245  |
| 6M Mortality    | True     | 2,387  |
|                 | False    | 18,545 |
|                 | Censored | 2,304  |
| 12M Mortality   | True     | 2,914  |
|                 | False    | 17,150 |
|                 | Censored | 3,172  |
| 1M Readmission  | True     | 856    |
|                 | False    | 20,764 |
|                 | Censored | 1,616  |
| 6M Readmission  | True     | 2,184  |
|                 | False    | 17,947 |
|                 | Censored | 3,105  |
| 12M Readmission | True     | 2,825  |
|                 | False    | 16,247 |
|                 | Censored | 4,164  |
| 12M PH          | True     | 2,724  |
|                 | False    | 16,497 |
|                 | Censored | 4,015  |

---

## 2. Structured EHR & Demographics

**Script:** `Data/01_build_INSPECT_structured_EHR_and_demographics_analysis.py`

### Steps:

* Construct long-format EHR using four domains: `condition`, `procedure`, `measurement`, `drug`.
* Merge in demographics and label outcomes.
* Compute age and categorize into bins.

**Output File:**

* `structured_ehr_all.csv.gz`

### Dataset Stats

### Table 2: Demographics

**By Gender**

| Gender | Count (%)      |
| ------ | -------------- |
| Female | 12,536 (55.8%) |
| Male   | 9,944 (44.2%)  |

**By Age Group**

| Age Bin | Count (%)      |
| ------- | -------------- |
| 18-39   | 3,569 (15.9%)  |
| 39-69   | 12,239 (54.4%) |
| 69-88   | 6,592 (29.3%)  |
| >88     | 80 (0.4%)      |

**By Race**

| Race                             | Count (%)      |
| -------------------------------- | -------------- |
| White                            | 12,190 (54.2%) |
| Asian                            | 3,488 (15.5%)  |
| Black or African American        | 1,419 (6.3%)   |
| Native Hawaiian or Other Pacific | 448 (2.0%)     |
| American Indian or Alaska Native | 85 (0.4%)      |
| No matching concept              | 4,850 (21.6%)  |

**By Ethnicity**

| Ethnicity              | Count (%)      |
| ---------------------- | -------------- |
| Not Hispanic or Latino | 17,992 (80.0%) |
| Hispanic or Latino     | 3,692 (16.4%)  |
| No matching concept    | 796 (3.5%)     |

---

## 3. Merging Radiology Impressions with Labels

**Script:** `Data/02_merge_radiology_impressions_with_labels.py`

### Steps:

* Merge radiology impressions (from `impressions_20250611.tsv`) with cohort data.
* Output a single unified file with text + structured labels.

**Output File:**

* `radiology_impressions_with_all_labels.csv.gz`

### Result Summary

| Metric              | Value  |
| ------------------- | ------ |
| Rows                | 23,248 |
| Unique Impressions  | 23,248 |
| Missing Impressions | 0      |

---

## File Summary

| File                                           | Description                   |
| ---------------------------------------------- | ----------------------------- |
| `INSPECT_full_dataset_with_impressions.csv.gz` | Labeled cohort with text      |
| `structured_ehr_all.csv.gz`                    | Long-format EHR + demo/labels |
| `radiology_impressions_with_all_labels.csv.gz` | Text + outcome labels         |

