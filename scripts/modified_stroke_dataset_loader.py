# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class StrokeDataset(Dataset):
    def __init__(self, csv_file, root_dir, clinical_features):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.clinical_features = clinical_features  # АэИз ['age', 'sex', 'blood_pressure', ...]

    def __len__(self):
      
        return len(self.data)

    def __getitem__(self, idx):
        
        row = self.data.iloc[idx]
        tabular_data = torch.tensor(row[self.clinical_features].values, dtype=torch.float32)
        ct_pre = np.load(os.path.join(self.root_dir, row['ct_pre_path']))
        ct_post = np.load(os.path.join(self.root_dir, row['ct_post_path']))
        mri = np.load(os.path.join(self.root_dir, row['mri_path']))
        label = torch.tensor(row['nihss_score'], dtype=torch.float32)
        return tabular_data, ct_pre, ct_post, mri, label