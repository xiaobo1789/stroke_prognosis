# 新增 data_loader.py
import nibabel as nib
import pandas as pd
from torch.utils.data import Dataset

class StrokeDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 加载CT/MRI数据
        ct_pre = nib.load(f"{self.img_dir}/{self.df.iloc[idx]['ct_pre_path']}").get_fdata()
        ct_post = nib.load("F:\GGSJJAIDS\image")
        mr_post = nib.load("F:\GGSJJAIDS\image")
        
        # 加载表格数据
        tabular = self.df.iloc[idx][['age', 'nihss_base', ...]].values
        
        # 加载标签 (NIHSS评分等级)
        label = self.df.iloc[idx]['nihss_category']  # 0-3等级
        
        if self.transform:
            ct_pre = self.transform(ct_pre)
            ...
        
        return {
            'ct_pre': ct_pre,
            'ct_post': ct_post,
            'mr_post': mr_post,
            'tabular': tabular,
            'label': label
        }