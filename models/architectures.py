import torch
import torch.nn as nn
from .daft import DAFT
from .resnet import ResNet10

class DAFTMultimodal(nn.Module):
    """
    基于DAFT的多模态中风预后预测模型
    """
    def __init__(self, tabular_dim, num_classes=2):
        super(DAFTMultimodal, self).__init__()
        
        # 图像分支 (ResNet-10)
        self.image_backbone = ResNet10(in_channels=6)
        
        # DAFT模块位置 (在最后一个残差块后)
        self.daft = DAFT(in_channels=512, tabular_dim=128)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 表格数据处理
        self.tabular_fc = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, image, tabular):
        # 处理表格数据
        tabular_processed = self.tabular_fc(tabular)
        
        # 图像特征提取
        x = self.image_backbone.conv1(image)
        x = self.image_backbone.bn1(x)
        x = self.image_backbone.relu(x)
        x = self.image_backbone.maxpool(x)
        
        x = self.image_backbone.layer1(x)
        x = self.image_backbone.layer2(x)
        x = self.image_backbone.layer3(x)
        x = self.image_backbone.layer4(x)  # [B, 512, D, H, W]
        
        # 应用DAFT
        x = self.daft(x, tabular_processed)
        
        # 分类
        out = self.classifier(x)
        
        return out  