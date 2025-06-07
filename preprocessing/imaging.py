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
    """����CT��עͼ���Ԥ������ˮ��"""
    def __init__(self, mode='train', input_size=(128, 128, 16)):
        self.mode = mode
        self.input_size = input_size
        
        # ����ת��
        base_transforms = [
            Resize(input_size),
            ScaleIntensity(minv=0.0, maxv=1.0)
        ]
        
        # ѵ��ʱ��������ǿ
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
        """������CT��עɨ��"""
        # ת��Ϊ����
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        # ���ͨ��ά�� [C, D, H, W]
        image_tensor = image_tensor.unsqueeze(0)
        
        # Ӧ��ת��
        transformed = self.transform(image_tensor)
        
        return transformed