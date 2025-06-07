import torch
import numpy as np
from monai.transforms import (
    RandRotate,
    RandZoom,
    RandGaussianNoise,
    Rand3DElastic,
    RandAdjustContrast,
    Compose
)

class CTPerfusionAugmentor:
    def __init__(self, apply_augmentation=True):
        self.apply_augmentation = apply_augmentation
        self.transforms = Compose([
            RandRotate(range_x=0, range_y=0, range_z=1, prob=0.5, key='image'),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.4, key='image'),
            # 其他变换添加 key='image'
        ])
    
    def __call__(self, data):
        if not self.apply_augmentation:
            return data
        return self.transforms(data)  # 输入为 {'image': img_tensor, ...}