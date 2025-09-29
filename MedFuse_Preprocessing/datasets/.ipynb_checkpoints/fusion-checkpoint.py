import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset 
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

R_CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']

CLASSES = [
       'Acute and unspecified renal failure', 'Acute cerebrovascular disease',
       'Acute myocardial infarction', 'Cardiac dysrhythmias',
       'Chronic kidney disease',
       'Chronic obstructive pulmonary disease and bronchiectasis',
       'Complications of surgical procedures or medical care',
       'Conduction disorders', 'Congestive heart failure; nonhypertensive',
       'Coronary atherosclerosis and other heart disease',
       'Diabetes mellitus with complications',
       'Diabetes mellitus without complication',
       'Disorders of lipid metabolism', 'Essential hypertension',
       'Fluid and electrolyte disorders', 'Gastrointestinal hemorrhage',
       'Hypertension with complications and secondary hypertension',
       'Other liver diseases', 'Other lower respiratory disease',
       'Other upper respiratory disease',
       'Pleurisy; pneumothorax; pulmonary collapse',
       'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
       'Respiratory failure; insufficiency; arrest (adult)',
       'Septicemia (except in labor)', 'Shock'
    ]

class MIMIC_CXR_EHR(Dataset):
    def __init__(self, args, metadata_with_labels, ehr_ds, cxr_ds, split='train'):
        
        self.CLASSES = CLASSES
        if 'radiology' in args.labels_set:
            self.CLASSES = R_CLASSES
        
        self.metadata_with_labels = metadata_with_labels
        self.cxr_files_paired = self.metadata_with_labels.dicom_id.values
        self.ehr_files_paired = (self.metadata_with_labels['stay'].values)
        self.cxr_files_all = cxr_ds.filenames_loaded
        self.ehr_files_all = ehr_ds.names
        self.ehr_files_unpaired = list(set(self.ehr_files_all) - set(self.ehr_files_paired))
        self.ehr_ds = ehr_ds
        self.cxr_ds = cxr_ds
        self.args = args
        self.split = split
        self.data_ratio = self.args.data_ratio 
        if split=='test':
            self.data_ratio =  1.0
        elif split == 'val':
            self.data_ratio =  0.0


    def __getitem__(self, index):
        if self.args.data_pairs == 'paired_ehr_cxr':
            ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
            cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_paired[index]]
            return ehr_data, cxr_data, labels_ehr, labels_cxr
        elif self.args.data_pairs == 'paired_ehr':
            ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
            cxr_data, labels_cxr = None, None
            return ehr_data, cxr_data, labels_ehr, labels_cxr
        elif self.args.data_pairs == 'radiology':
            ehr_data, labels_ehr = np.zeros((1, 10)), np.zeros(self.args.num_classes)
            cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_all[index]]
            return ehr_data, cxr_data, labels_ehr, labels_cxr
        elif self.args.data_pairs == 'partial_ehr':
            ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_all[index]]
            cxr_data, labels_cxr = None, None
            return ehr_data, cxr_data, labels_ehr, labels_cxr
        
        elif self.args.data_pairs == 'partial_ehr_cxr':
            if index < len(self.ehr_files_paired):
                ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
                cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_paired[index]]
            else:
                index = random.randint(0, len(self.ehr_files_unpaired)-1) 
                ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_unpaired[index]]
                cxr_data, labels_cxr = None, None
            return ehr_data, cxr_data, labels_ehr, labels_cxr
    
    def __len__(self):
        if 'paired' in self.args.data_pairs:
            return len(self.ehr_files_paired)
        elif self.args.data_pairs == 'partial_ehr':
            return len(self.ehr_files_all)
        elif self.args.data_pairs == 'radiology':
            return len(self.cxr_files_all)
        elif self.args.data_pairs == 'partial_ehr_cxr':
            return len(self.ehr_files_paired) + int(self.data_ratio * len(self.ehr_files_unpaired)) 


def loadmetadata(args):

    data_dir = args.cxr_data_dir
    cxr_metadata = pd.read_csv(f'{data_dir}/mimic-cxr-2.0.0-metadata.csv')
    icu_stay_metadata = pd.read_csv(f'{args.ehr_data_dir}/all_stays.csv')
    columns = ['subject_id', 'stay_id', 'intime', 'outtime']
    
    # only common subjects with both icu stay and an xray
    cxr_merged_icustays = cxr_metadata.merge(icu_stay_metadata[columns ], how='inner', on='subject_id')
    
    # combine study date time
    cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
    cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(cxr_merged_icustays['StudyDate'].astype(str) + ' ' + cxr_merged_icustays['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")
    
    cxr_merged_icustays.intime=pd.to_datetime(cxr_merged_icustays.intime)
    cxr_merged_icustays.outtime=pd.to_datetime(cxr_merged_icustays.outtime)
    end_time = cxr_merged_icustays.outtime
    if args.task == 'in-hospital-mortality':
        end_time = cxr_merged_icustays.intime + pd.DateOffset(hours=48)

    cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&((cxr_merged_icustays.StudyDateTime<=end_time))]

    # cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&((cxr_merged_icustays.StudyDateTime<=cxr_merged_icustays.outtime))]
    # select cxrs with the ViewPosition == 'AP
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']

    groups = cxr_merged_icustays_AP.groupby('stay_id')
    print("Final dicom_id in merge cxr:", len(cxr_merged_icustays_AP["dicom_id"]))
    cxr_merged_icustays_AP.to_csv("cxr_debug_pheno.csv", index=False)

    groups_selected = []
    for group in groups:
        # select the latest cxr for the icu stay
        selected = group[1].sort_values('StudyDateTime').tail(1).reset_index()
        groups_selected.append(selected)
    groups = pd.concat(groups_selected, ignore_index=True)
    # import pdb; pdb.set_trace()

    # groups['cxr_length'] = (groups['StudyDateTime'] - groups['intime']).astype('timedelta64[h]')
    return groups

# def 
def load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds):

    cxr_merged_icustays = loadmetadata(args) 

    # cxr_merged_icustays['cxr_length'] = (cxr_merged_icustays['StudyDateTime'] - cxr_merged_icustays['intime'] ).astype('timedelta64[h]')

    # import pdb; pdb.set_trace()

    splits_labels_train = pd.read_csv(f'{args.ehr_data_dir}/{args.task}/train_listfile.csv')
    splits_labels_val = pd.read_csv(f'{args.ehr_data_dir}/{args.task}/val_listfile.csv')
    splits_labels_test = pd.read_csv(f'{args.ehr_data_dir}/{args.task}/test_listfile.csv')
    
    print(f"train={len(splits_labels_train)}, val={len(splits_labels_val)}, test={len(splits_labels_test)}, total={len(splits_labels_train)+len(splits_labels_val)+len(splits_labels_test)}")

    train_meta_with_labels = cxr_merged_icustays.merge(splits_labels_train, how='inner', on='stay_id')
    val_meta_with_labels = cxr_merged_icustays.merge(splits_labels_val, how='inner', on='stay_id')
    test_meta_with_labels = cxr_merged_icustays.merge(splits_labels_test, how='inner', on='stay_id')

    # Paired csv output    
    if args.task == 'in-hospital-mortality':
        train_meta_with_labels.to_csv("train_metadata_mortality.csv", index=False)
        val_meta_with_labels.to_csv("val_metadata_mortality.csv", index=False)
        test_meta_with_labels.to_csv("test_metadata_mortality.csv", index=False)
        else:
            train_meta_with_labels.to_csv("train_metadata_pheno.csv", index=False)
            val_meta_with_labels.to_csv("val_metadata_pheno.csv", index=False)
            test_meta_with_labels.to_csv("test_metadata_pheno.csv", index=False)
    
    train_ds = MIMIC_CXR_EHR(args, train_meta_with_labels, ehr_train_ds, cxr_train_ds)
    val_ds = MIMIC_CXR_EHR(args, val_meta_with_labels, ehr_val_ds, cxr_val_ds, split='val')
    test_ds = MIMIC_CXR_EHR(args, test_meta_with_labels, ehr_test_ds, cxr_test_ds, split='test')
    
    # Partial csv output
    
    if args.task == 'in-hospital-mortality':
        manifest_partial_ehr_cxr(train_ds, "train", listfile_df=splits_labels_train, base="metadata_partial_mortality", all_name="partial_ehr_cxr_all_mortality.csv")
        manifest_partial_ehr_cxr(val_ds,   "val",  listfile_df=splits_labels_val, base="metadata_partial_mortality", all_name="partial_ehr_cxr_all_mortality.csv")
        manifest_partial_ehr_cxr(test_ds,  "test", listfile_df=splits_labels_test, base="metadata_partial_mortality", all_name="partial_ehr_cxr_all_mortality.csv")
        else:
            manifest_partial_ehr_cxr(train_ds, "train", listfile_df=splits_labels_train)
            manifest_partial_ehr_cxr(val_ds,   "val",  listfile_df=splits_labels_val)
            manifest_partial_ehr_cxr(test_ds,  "test", listfile_df=splits_labels_test)

#     printPrevalence(train_meta_with_labels, args)
#     printPrevalence(val_meta_with_labels, args)
#     printPrevalence(test_meta_with_labels, args)

#     printPrevalence(splits_labels_train, args)
#     printPrevalence(splits_labels_val, args)
#     printPrevalence(splits_labels_test, args)


    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)

    return train_dl, val_dl, test_dl

def printPrevalence(merged_file, args):
    if args.labels_set == 'pheno':
        total_rows = len(merged_file)
        print(merged_file[CLASSES].sum()/total_rows)
    else:
        total_rows = len(merged_file)
        print(merged_file['y_true'].value_counts())
    # import pdb; pdb.set_trace()

def my_collate(batch):
    x = [item[0] for item in batch]
    pairs = [False if item[1] is None else True for item in batch]
    img = torch.stack([torch.zeros(3, 224, 224) if item[1] is None else item[1] for item in batch])
    x, seq_length = pad_zeros(x)
    targets_ehr = np.array([item[2] for item in batch])
    targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
    return [x, img, targets_ehr, targets_cxr, seq_length, pairs]

def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length



import os
import random
import numpy as np
import pandas as pd
from typing import Optional

def _first_df_with_key(obj, key: str) -> Optional[pd.DataFrame]:
    for _, val in vars(obj).items():
        if isinstance(val, pd.DataFrame) and key in val.columns:
            return val
    return None

def _maybe_add_path_column(obj, key: str, out_key: str) -> Optional[pd.DataFrame]:
    candidates = ["path_by_id", "paths", "filepaths", "id_to_path", "filepath_by_id"]
    for name in candidates:
        if hasattr(obj, name):
            mapping = getattr(obj, name)
            if isinstance(mapping, dict) and len(mapping) > 0:
                return pd.DataFrame({key: list(mapping.keys()), out_key: list(mapping.values())})
    return None

def _left_merge_fill(df_left: pd.DataFrame, df_right: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_right is None or not isinstance(df_right, pd.DataFrame) or df_right.empty:
        return df_left

    preferred = ["stay", "dicom_id"]
    common_keys = [k for k in preferred if k in df_left.columns and k in df_right.columns]
    if not common_keys:
        inter = list(set(df_left.columns) & set(df_right.columns))
        if not inter:
            return df_left
        common_keys = [inter[0]]

    keep_cols = [c for c in df_right.columns if not df_right[c].isna().all()]
    right_small = df_right[list(dict.fromkeys(common_keys + keep_cols))].drop_duplicates()

    merged = df_left.merge(right_small, on=common_keys, how="left", suffixes=("", "_r"))

    for c in right_small.columns:
        if c in common_keys:
            continue
        if c in df_left.columns:
            src = c + "_r" if (c + "_r") in merged.columns else c
            if src in merged.columns:
                merged[c] = merged[c].where(~merged[c].isna(), merged[src])
                if src in merged.columns and src != c:
                    merged = merged.drop(columns=[src])
        else:
            src = c + "_r" if (c + "_r") in merged.columns else c
            if src in merged.columns and src != c:
                merged.rename(columns={src: c}, inplace=True)
    return merged

def manifest_partial_ehr_cxr(
    ds,
    split_name,                 # "train" / "val" / "test"
    seed=42,
    save_dir=".",
    base="metadata_partial_pheno",
    all_name="partial_ehr_cxr_all_pheno.csv",
    reset_all=True,
    listfile_df: Optional[pd.DataFrame] = None,
):
    assert ds.args.data_pairs == "partial_ehr_cxr"
    os.makedirs(save_dir, exist_ok=True)
    rng = random.Random(seed)

    paired = pd.DataFrame({
        "split": split_name,
        "pair_status": "paired",
        "stay": ds.ehr_files_paired,
        "dicom_id": ds.cxr_files_paired
    })

    k = int(ds.data_ratio * len(ds.ehr_files_unpaired))
    unpaired_list = list(ds.ehr_files_unpaired)
    rng.shuffle(unpaired_list)
    unpaired_sample = unpaired_list[:k]
    unpaired = pd.DataFrame({
        "split": split_name,
        "pair_status": "ehr_unpaired",
        "stay": unpaired_sample,
        "dicom_id": [np.nan] * len(unpaired_sample)
    })

    df = pd.concat([paired, unpaired], ignore_index=True)

    meta = ds.metadata_with_labels
    label_cols = [c for c in getattr(ds, "CLASSES", []) if c in meta.columns]

    for c in label_cols:
        if c not in df.columns:
            df[c] = np.nan

    if label_cols and "dicom_id" in meta.columns:
        cols_avail = ["dicom_id"] + label_cols
        meta_d = meta[cols_avail].drop_duplicates()
        tmp = df[["dicom_id"]].merge(meta_d, on="dicom_id", how="left")
        for c in label_cols:
            df[c] = df[c].fillna(tmp[c])

    ehr_key_in_meta = "stay" if "stay" in meta.columns else ("stay_id" if "stay_id" in meta.columns else None)
    if label_cols and ehr_key_in_meta:
        cols_avail = [ehr_key_in_meta] + label_cols
        meta_e = meta[cols_avail].drop_duplicates()
        tmp = df[["stay"]].merge(meta_e, left_on="stay", right_on=ehr_key_in_meta, how="left")
        for c in label_cols:
            df[c] = df[c].fillna(tmp[c])

    drop_keys = set(label_cols + ["dicom_id"])
    if ehr_key_in_meta:
        drop_keys.add(ehr_key_in_meta)
    extra_cols_all = [c for c in meta.columns if c not in drop_keys and c != "index"]

    for c in extra_cols_all:
        if c not in df.columns:
            df[c] = np.nan

    if "dicom_id" in meta.columns and extra_cols_all:
        cols_avail = ["dicom_id"] + [c for c in extra_cols_all if c in meta.columns]
        tmp = df[["dicom_id"]].merge(meta[cols_avail].drop_duplicates(), on="dicom_id", how="left")
        for c in cols_avail[1:]:
            df[c] = df[c].fillna(tmp[c])

    if ehr_key_in_meta and extra_cols_all:
        cols_avail = [ehr_key_in_meta] + [c for c in extra_cols_all if c in meta.columns]
        tmp = df[["stay"]].merge(meta[cols_avail].drop_duplicates(), left_on="stay", right_on=ehr_key_in_meta, how="left")
        for c in cols_avail[1:]:
            df[c] = df[c].fillna(tmp[c])

    ehr_info_df = _first_df_with_key(ds.ehr_ds, "stay")
    if ehr_info_df is not None:
        ehr_use = [c for c in ehr_info_df.columns if c != "stay"]
        if ehr_use:
            df = df.merge(ehr_info_df[["stay"] + ehr_use].drop_duplicates(), on="stay", how="left")

    ehr_paths_df = _maybe_add_path_column(ds.ehr_ds, key="stay", out_key="ehr_path")
    if ehr_paths_df is not None:
        df = df.merge(ehr_paths_df, on="stay", how="left")

    cxr_info_df = _first_df_with_key(ds.cxr_ds, "dicom_id")
    if cxr_info_df is not None:
        cxr_use = [c for c in cxr_info_df.columns if c != "dicom_id"]
        if cxr_use:
            df = df.merge(cxr_info_df[["dicom_id"] + cxr_use].drop_duplicates(), on="dicom_id", how="left")

    cxr_paths_df = _maybe_add_path_column(ds.cxr_ds, key="dicom_id", out_key="cxr_path")
    if cxr_paths_df is not None:
        df = df.merge(cxr_paths_df, on="dicom_id", how="left")

    df = _left_merge_fill(df, listfile_df)

    split_path = os.path.join(save_dir, f"{split_name}_{base}.csv")
    df.to_csv(split_path, index=False)

    all_path = os.path.join(save_dir, all_name)
    if reset_all and os.path.exists(all_path):
        os.remove(all_path)
    write_header = not os.path.exists(all_path)
    df.to_csv(all_path, mode="a", header=write_header, index=False)

    print(f"✅ Saved split CSV: {split_path}  shape={df.shape}")
    print(f"✅ Appended to ALL CSV: {all_path}")
    return df
