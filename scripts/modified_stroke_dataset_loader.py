import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class StrokeDataset(Dataset):
    def __init__(self, csv_file, root_dir, clinical_features):
        """
        ��ʼ�����ݼ���
        ������
            csv_file (str): CSV�ļ�·������'data/train.csv'����
            root_dir (str): ���ݸ�Ŀ¼����'data/'����
            clinical_features (list): �ٴ����������б���configs/base_config.py��ȡ��
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.clinical_features = clinical_features  # ���� ['age', 'sex', 'blood_pressure', ...]

    def __len__(self):
        """�������ݼ���С��"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        ��ȡ����������
        ���أ�
            tabular_data (torch.Tensor): ������ݡ�
            ct_pre (np.ndarray): ����ǰCTƽɨ���ݡ�
            ct_post (np.ndarray): ���ƺ�CTƽɨ���ݡ�
            mri (np.ndarray): MRI���ݡ�
            label (torch.Tensor): NIHSS���֡�
        """
        row = self.data.iloc[idx]
        tabular_data = torch.tensor(row[self.clinical_features].values, dtype=torch.float32)
        ct_pre = np.load(os.path.join(self.root_dir, row['ct_pre_path']))
        ct_post = np.load(os.path.join(self.root_dir, row['ct_post_path']))
        mri = np.load(os.path.join(self.root_dir, row['mri_path']))
        label = torch.tensor(row['nihss_score'], dtype=torch.float32)
        return tabular_data, ct_pre, ct_post, mri, label