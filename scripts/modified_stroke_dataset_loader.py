import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class StrokeDataset(Dataset):
    def __init__(self, csv_file, root_dir, clinical_features):
        """
        初始化数据集。
        参数：
            csv_file (str): CSV文件路径（如'data/train.csv'）。
            root_dir (str): 数据根目录（如'data/'）。
            clinical_features (list): 临床特征列名列表，从configs/base_config.py获取。
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.clinical_features = clinical_features  # 例如 ['age', 'sex', 'blood_pressure', ...]

    def __len__(self):
        """返回数据集大小。"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个样本。
        返回：
            tabular_data (torch.Tensor): 表格数据。
            ct_pre (np.ndarray): 治疗前CT平扫数据。
            ct_post (np.ndarray): 治疗后CT平扫数据。
            mri (np.ndarray): MRI数据。
            label (torch.Tensor): NIHSS评分。
        """
        row = self.data.iloc[idx]
        tabular_data = torch.tensor(row[self.clinical_features].values, dtype=torch.float32)
        ct_pre = np.load(os.path.join(self.root_dir, row['ct_pre_path']))
        ct_post = np.load(os.path.join(self.root_dir, row['ct_post_path']))
        mri = np.load(os.path.join(self.root_dir, row['mri_path']))
        label = torch.tensor(row['nihss_score'], dtype=torch.float32)
        return tabular_data, ct_pre, ct_post, mri, label