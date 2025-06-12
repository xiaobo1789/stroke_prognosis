import torch
import torch.nn as nn
from .daft import DAFT
from .resnet import ResNet10

class DAFTMultimodal(nn.Module):
    """
    ����DAFT�Ķ�ģ̬�з�Ԥ��Ԥ��ģ��
    """
    def __init__(self, tabular_dim, num_classes=2):
        super(DAFTMultimodal, self).__init__()
        
        # ͼ���֧ (ResNet-10)
        self.image_backbone = ResNet10(in_channels=6)
        
        # DAFTģ��λ�� (�����һ���в���)
        self.daft = DAFT(in_channels=512, tabular_dim=128)
        
        # ����ͷ
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # ������ݴ���
        self.tabular_fc = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, image, tabular):
        # ����������
        tabular_processed = self.tabular_fc(tabular)
        
        # ͼ��������ȡ
        x = self.image_backbone.conv1(image)
        x = self.image_backbone.bn1(x)
        x = self.image_backbone.relu(x)
        x = self.image_backbone.maxpool(x)
        
        x = self.image_backbone.layer1(x)
        x = self.image_backbone.layer2(x)
        x = self.image_backbone.layer3(x)
        x = self.image_backbone.layer4(x)  # [B, 512, D, H, W]
        
        # Ӧ��DAFT
        x = self.daft(x, tabular_processed)
        
        # ����
        out = self.classifier(x)
        
        return out  