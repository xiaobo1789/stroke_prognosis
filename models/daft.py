import torch
import torch.nn as nn

class DAFTBlock(nn.Module):
    """动态仿射特征变换模块"""
    def __init__(self, img_channels, tabular_dim, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(img_channels + tabular_dim, (img_channels + tabular_dim) // reduction),
            nn.ReLU(),
            nn.Linear((img_channels + tabular_dim) // reduction, img_channels * 2)
        )

    def forward(self, img_feats, tabular):
        pooled = self.gap(img_feats).flatten(1)
        combined = torch.cat([pooled, tabular], dim=1)
        params = self.fc(combined)
        scale, shift = params.chunk(2, dim=1)
        
        # 扩展维度匹配特征图 [B, C, 1, 1, 1]
        scale = scale.view(*scale.shape, 1, 1, 1)
        shift = shift.view(*shift.shape, 1, 1, 1)
        
        return img_feats * scale + shift
class DAFT(nn.Module):
    """Dynamic Affine Feature Map Transform (DAFT)模块"""
    def __init__(self, base_resnet, tabular_dim, num_classes=1, bottleneck_factor=5):
        super().__init__()
        self.resnet = base_resnet
        self.resnet.fc = nn.Identity()
        # 通过前向传播获取最后一个残差块的通道数（或直接访问 resnet 结构）
        dummy_input = torch.randn(1, 6, 128, 128, 16)  # 假设输入尺寸
        with torch.no_grad():
            features = self.resnet.layer3(dummy_input)  # 假设 DAFT 插入在 layer3 后
        in_channels = features.shape[1]  # 动态获取通道数
        self.daft = DAFT(in_channels, tabular_dim, bottleneck_factor=bottleneck_factor)
        self.fc = nn.Linear(in_channels, num_classes)
    def forward(self, image_feats, tabular_data):
        """
        image_feats: 图像特征 [B, C, D, H, W]
        tabular_data: 表格数据 [B, tabular_dim]
        """
        # 全局平均池化
        gap = torch.mean(image_feats, dim=(2, 3, 4))  # [B, C]
        
        # 拼接特征
        combined = torch.cat([gap, tabular_data], dim=1)  # [B, C + tabular_dim]
        
        # 通过瓶颈层
        bottleneck_out = self.bottleneck(combined)  # [B, bottleneck_dim]
        
        # 计算缩放和偏移
        scale = self.scale(bottleneck_out)  # [B, C]
        shift = self.shift(bottleneck_out)  # [B, C]
        
        # 调整形状以匹配图像特征
        scale = scale.view(*scale.shape, 1, 1, 1)  # [B, C, 1, 1, 1]
        shift = shift.view(*shift.shape, 1, 1, 1)  # [B, C, 1, 1, 1]
        
        # 应用仿射变换
        transformed_feats = scale * image_feats + shift
        
        return transformed_feats

class DAFTResNet(nn.Module):
    def __init__(self, base_resnet, tabular_dim, num_classes=1, bottleneck_factor=5):  # 添加 bottleneck_factor
        super(DAFTResNet, self).__init__()
        self.resnet = base_resnet
        self.resnet.fc = nn.Identity()
        in_channels = 256  # 假设最后一个残差块通道数为256
        self.daft = DAFT(in_channels, tabular_dim, bottleneck_factor=bottleneck_factor)  # 传递参数
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x_img, x_tab):
        # 通过ResNet提取特征
        x = self.resnet.conv1(x_img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)  # [B, 256, D, H, W]
        
        # 应用DAFT变换
        x = self.daft(x, x_tab)
        
        # 继续通过ResNet
        x = self.resnet.layer4(x)
        
        # 全局平均池化和分类
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x