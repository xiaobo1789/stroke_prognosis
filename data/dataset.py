import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data.preprocessing import TabularPreprocessor, CTPerfusionPreprocessor

class StrokeDataset(Dataset):
    def __init__(self, csv_path, ct_root_dir, is_train=True, target_shape=(128, 128, 16)):
        """
        csv_path: 包含表格数据和标签的CSV文件路径
        ct_root_dir: 存放CT灌注扫描数据的根目录
        is_train: 是否为训练集
        target_shape: 目标CT图像尺寸
        """
        self.df = pd.read_csv(csv_path)
        self.ct_root_dir = ct_root_dir
        self.target_shape = target_shape
        self.is_train = is_train
        
        # 初始化预处理器
        self.tabular_preprocessor = TabularPreprocessor()
        self.ct_preprocessor = CTPerfusionPreprocessor(target_shape)
        
        # 预处理表格数据
        tabular_features = self.df.drop(['patient_id', 'mrs_score', 'label'], axis=1)
        if is_train:
            self.tabular_data = self.tabular_preprocessor.fit_transform(tabular_features.values)
        else:
            self.tabular_data = self.tabular_preprocessor.transform(tabular_features.values)
        
        # 标签：二分类（0: mRS 0-2, 1: mRS 3-6）
        self.labels = self.df['nihss_category'].values
        
        # 患者ID列表
        self.patient_ids = self.df['patient_id'].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 获取表格数据
        tabular = self.tabular_data[idx].astype(np.float32)
        
        # 获取CT灌注图像
        patient_id = self.patient_ids[idx]
        ct_path = os.path.join(self.ct_root_dir, f"{patient_id}.npy")
        ct_post_path = os.path.join(self.ct_root_dir, "post", f"{patient_id}.npy")  # 假设治疗后CT在post子目录
        ct_post = np.load(ct_post_path) if os.path.exists(ct_post_path) else np.zeros(self.target_shape)
        ct_post_processed = self.ct_preprocessor.process(ct_post)
        # 加载CT扫描数据
        if os.path.exists(ct_path):
            ct_scan = np.load(ct_path)
        else:
            # 如果文件不存在，创建空数组
            ct_scan = np.zeros((50, 512, 512), dtype=np.float32)
        
        # 预处理CT扫描
        ct_processed = self.ct_preprocessor.process(ct_scan)
        
        # 转换为张量
        tabular_tensor = torch.tensor(tabular, dtype=torch.float32)
        ct_tensor = torch.tensor(ct_processed, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return {
            'patient_id': patient_id,
            'tabular': tabular_tensor,
            'image_pre': ct_tensor,  # 治疗前CT
            'image_post': ct_post_processed,  # 治疗后CT
            'label': label  # NIHSS分级标签（0-3）
        }