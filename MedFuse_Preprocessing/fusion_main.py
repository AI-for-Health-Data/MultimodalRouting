from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
from pathlib import Path
import torch

from ehr_utils.preprocessing import Discretizer, Normalizer
from datasets.ehr_dataset import get_datasets
from datasets.cxr_dataset import get_cxr_datasets
from datasets.fusion import load_cxr_ehr
from arguments import args_parser

parser = args_parser()
args = parser.parse_args()
print(args)

path = Path(args.save_dir)
path.mkdir(parents=True, exist_ok=True)

seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)


def read_timeseries(args):
    path = f'{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)


discretizer = Discretizer(
    timestep=float(args.timestep),
    store_masks=True,
    impute_strategy='previous',
    start_time='zero'
)

discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = f'normalizers/ph_ts{args.timestep}.input_str:previous.start_time:zero.normalizer'
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)
cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)

train_dl, val_dl, test_dl = load_cxr_ehr(
    args,
    ehr_train_ds,
    ehr_val_ds,
    cxr_train_ds,
    cxr_val_ds,
    ehr_test_ds,
    cxr_test_ds
)

print("Train samples:", len(train_dl.dataset))
print("Val samples:", len(val_dl.dataset))
print("Test samples:", len(test_dl.dataset))
