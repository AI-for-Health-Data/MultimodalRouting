import os
import re
import pandas as pd
import numpy as np
from datetime import timedelta
from transformers import AutoTokenizer

def preprocess_notes(text):
    y = re.sub(r'\[(.*?)\]', '', str(text))
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'\s+', ' ', y)
    return y.strip().lower()

def split_and_expand(df, text_col, tokenizer, max_length=512, stride=0):
    chunked = df[text_col].fillna('').apply(
        lambda t: tokenizer.encode(t, add_special_tokens=False)
    ).apply(
        lambda ids: [ids[i:i + max_length] for i in range(0, len(ids), max_length - stride or max_length)]
    )
    max_chunks = chunked.map(len).max()
    note_cols = []
    for i in range(max_chunks):
        col = f'note_{i}'
        df[col] = chunked.map(lambda chunks: tokenizer.decode(chunks[i], clean_up_tokenization_spaces=True) if i < len(chunks) else "")
        note_cols.append(col)
    return df, note_cols

def get_readmission_flag(df):
    df = df.sort_values(['subject_id', 'intime'])
    df['next_intime'] = df.groupby('subject_id')['intime'].shift(-1)
    df['readmit_30d'] = (
        (df['next_intime'] - df['outtime']) <= pd.Timedelta(days=30)
    ).fillna(False).astype(int)
    return df[['subject_id', 'hadm_id', 'stay_id', 'readmit_30d']]

def categorize_race(race_val):
    r = str(race_val).upper()
    if 'HISPANIC' in r or 'LATINO' in r or 'SOUTH AMERICAN' in r or 'PORTUGUESE' in r or 'BRAZILIAN' in r:
        return 'Hispanic'
    if 'BLACK' in r or 'AFRICAN' in r or 'CAPE VERDEAN' in r or 'CARIBBEAN ISLAND' in r or 'HAITIAN' in r:
        return 'Black'
    if 'ASIAN' in r or 'KOREAN' in r or 'CHINESE' in r or 'INDIAN' in r or 'SOUTH EAST ASIAN' in r:
        return 'Asian'
    if 'WHITE' in r or 'EASTERN EUROPEAN' in r or 'RUSSIAN' in r or 'OTHER EUROPEAN' in r:
        return 'White'
    if ('AMERICAN INDIAN' in r or 'ALASKA NATIVE' in r or 'NATIVE HAWAIIAN' in r or 
        'PACIFIC ISLANDER' in r or 'MIDDLE EASTERN' in r or 'MULTIPLE' in r or
        r in {'OTHER', 'PATIENT DECLINED TO ANSWER', 'UNABLE TO OBTAIN', 'UNKNOWN', 'UNKNOWN/NOT SPECIFIED'}):
        return 'Other'
    return 'Other'

def categorize_ethnicity(race_val):
    r = str(race_val).upper()
    return 'Hispanic' if ('HISPANIC' in r or 'LATINO' in r) else 'Non-Hispanic'

def categorize_insurance(ins):
    if pd.isna(ins):
        return 'Other'
    i = str(ins).strip().lower()
    if i == 'medicare':
        return 'Medicare'
    if i == 'medicaid':
        return 'Medicaid'
    if i == 'private':
        return 'Private'
    return 'Other'

def categorize_age(age):
    if pd.isna(age):
        return 'Other'
    if 18 <= age <= 29: return '18-29'
    elif 30 <= age <= 49: return '30-49'
    elif 50 <= age <= 69: return '50-69'
    elif 70 <= age <= 90: return '70-90'
    elif age > 90: return '90+'
    else: return 'Other'

admissions = pd.read_csv(
    'admissions.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','admittime','dischtime','insurance','race'],
    parse_dates=['admittime','dischtime']
)
patients = pd.read_csv(
    'patients.csv.gz', compression='gzip',
    usecols=['subject_id','gender','anchor_age']
)
icustays = pd.read_csv(
    'icustays.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','stay_id','intime','outtime'],
    parse_dates=['intime','outtime']
)
labevents = pd.read_csv(
    'labevents.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','itemid','charttime','valuenum'],
    parse_dates=['charttime'], low_memory=False
)
notes = pd.read_csv(
    'discharge.csv.gz', compression='gzip',
    usecols=['subject_id','hadm_id','charttime','text'],
    parse_dates=['charttime'], low_memory=False
)

stays = icustays.merge(patients, on='subject_id', how='left')
stays = stays[stays['anchor_age'] >= 18]
stays['dur_h'] = (stays['outtime'] - stays['intime']).dt.total_seconds() / 3600
stays = stays[stays['dur_h'] >= 30]

readmit_flags = get_readmission_flag(stays)
first_icu = stays.sort_values('intime').drop_duplicates('subject_id', keep='first')
cohort = first_icu.merge(
    readmit_flags,
    on=['subject_id','hadm_id','stay_id'],
    how='left'
).fillna({'readmit_30d': 0})

demo = admissions[['subject_id','hadm_id','race','insurance']]
demo['race_cat']      = demo['race'].apply(categorize_race)
demo['ethnicity_cat'] = demo['race'].apply(categorize_ethnicity)
demo['insurance_cat'] = demo['insurance'].apply(categorize_insurance)
cohort = cohort.merge(demo, on=['subject_id','hadm_id'], how='left')
cohort['age_bucket'] = cohort['anchor_age'].apply(categorize_age)
cohort['gender'] = cohort['gender'].str.lower().map(lambda x: 'female' if x == 'f' else ('male' if x == 'm' else 'other'))


data = cohort[['subject_id','hadm_id','stay_id','outtime']]
labs = labevents.merge(data, on=['subject_id','hadm_id'], how='inner')
labs['delta_h'] = (labs['outtime'] - labs['charttime']).dt.total_seconds() / 3600
labs = labs[labs['delta_h'].between(0, 24)]
labs['hour_bin'] = (labs['delta_h'] // 2).astype(int)
labs['lab_col'] = labs.apply(lambda r: f"lab_{r['itemid']}_b{r['hour_bin']}", axis=1)
lab_agg = labs.pivot_table(
    index=['subject_id','hadm_id','stay_id'],
    columns='lab_col',
    values='valuenum',
    aggfunc='mean'
).reset_index()
structured = cohort.merge(lab_agg, on=['subject_id','hadm_id','stay_id'], how='left')

demographic_cols = ['subject_id', 'hadm_id', 'stay_id', 'readmit_30d', 'anchor_age', 'age_bucket', 'gender', 'race_cat', 'ethnicity_cat', 'insurance_cat']
other_cols = [c for c in structured.columns if c not in demographic_cols]
structured = structured[demographic_cols + other_cols]

structured.to_csv('structured_dataset.csv', index=False)
print("Structured final shape:", structured.shape)
print("Structured readmission positives:", structured['readmit_30d'].sum())

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', use_fast=True)

notes24 = notes.merge(data, on=['subject_id','hadm_id'], how='inner')
notes24['delta_h'] = (notes24['outtime'] - notes24['charttime']).dt.total_seconds() / 3600
notes24 = notes24[notes24['delta_h'].between(0, 24)]

agg_notes = (
    notes24.groupby(['subject_id','hadm_id'])['text']
           .apply(lambda s: ' '.join(s.astype(str)))
           .reset_index()
)
agg_notes['cleaned_text'] = agg_notes['text'].apply(preprocess_notes)

cohort_notes = cohort.merge(
    agg_notes[['subject_id','hadm_id','cleaned_text']],
    on=['subject_id','hadm_id'],
    how='left'
)

cohort_expanded, note_cols = split_and_expand(
    cohort_notes, 'cleaned_text', tokenizer, max_length=512, stride=50
)

unstructured_cols = demographic_cols + note_cols
cohort_expanded[unstructured_cols].to_csv('unstructured_512token_notes.csv', index=False)
print("Unstructured final shape:", cohort_expanded[unstructured_cols].shape)
print(f"Created {len(note_cols)} note chunk columns per patient.")

common_ids = set(structured['subject_id']).intersection(cohort_expanded['subject_id'])
structured_common = structured[structured['subject_id'].isin(common_ids)].copy()
unstructured_common = cohort_expanded[cohort_expanded['subject_id'].isin(common_ids)].copy()

structured_common.to_csv('structured_common_subjects.csv', index=False)
unstructured_common.to_csv('unstructured_common_subjects.csv', index=False)

print("Structured (common IDs) shape:", structured_common.shape)
print("Structured readmission positives:", structured_common['readmit_30d'].sum())
print("Unstructured (common IDs) shape:", unstructured_common.shape)
print("Unstructured readmission positives:", unstructured_common['readmit_30d'].sum())

for col in ['age_bucket', 'ethnicity_cat', 'race_cat', 'insurance_cat', 'gender']:
    print(f"\nStructured {col} value counts:")
    print(structured_common[col].value_counts())
    print(f"Unstructured {col} value counts:")
    print(unstructured_common[col].value_counts())
