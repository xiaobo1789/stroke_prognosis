import numpy as np
import torch
import monai
from monai.transforms import (
    Compose,
    Resize,
    ScaleIntensity,
    RandRotate,
    RandZoom,
    RandGaussianNoise,
    Rand3DElastic,
    RandAdjustContrast
)

class CTPerfusionProcessor:
    """处理CT灌注图像的预处理流水线"""
    def __init__(self, mode='train', input_size=(128, 128, 16)):
        self.mode = mode
        self.input_size = input_size
        
        # 基础转换
        base_transforms = [
            Resize(input_size),
            ScaleIntensity(minv=0.0, maxv=1.0)
        ]
        
        # 训练时的数据增强
        if mode == 'train':
            base_transforms.extend([
                RandRotate(range_x=0, range_y=0, range_z=1, prob=0.5),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.4),
                RandGaussianNoise(mean=0, std=0.01, prob=0.5),
                Rand3DElastic(sigma_range=(5,7), magnitude_range=(50,140), 
                              padding_mode='zeros', prob=0.5),
                RandAdjustContrast(gamma=(1,1.5), prob=0.5)
            ])
        
        self.transform = Compose(base_transforms)
    
    def process(self, image):
        """处理单个CT灌注扫描"""
        # 转换为张量
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        # 添加通道维度 [C, D, H, W]
        image_tensor = image_tensor.unsqueeze(0)
        
        # 应用转换
        transformed = self.transform(image_tensor)
        
        return transformed