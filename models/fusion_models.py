import torch
import torch.nn as nn

class LateFusion(nn.Module):
    """晚期融合模型"""
    def __init__(self, tabular_model, image_model, num_classes=1):
        super(LateFusion, self).__init__()
        self.tabular_model = tabular_model
        self.image_model = image_model
        
        # 冻结特征提取器
        for param in self.tabular_model.parameters():
            param.requires_grad = False
        for param in self.image_model.parameters():
            param.requires_grad = False
        
        # 分类器
        tabular_out = 1  # TabNet输出为1维
        image_out = 1    # ResNet输出为1维
        self.fc = nn.Sequential(
            nn.Linear(tabular_out + image_out, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x_img, x_tab):
        # 通过两个模型
        tab_out = self.tabular_model(x_tab)
        img_out = self.image_model(x_img)
        
        # 拼接特征
        combined = torch.cat([tab_out, img_out], dim=1)
        out = self.fc(combined)
        return out

class HybridFusion(nn.Module):
    """混合融合模型"""
    def __init__(self, tabular_model, image_model, num_classes=1, hidden_dim=64):
        super(HybridFusion, self).__init__()
        self.tabular_model = tabular_model
        self.image_model = image_model
        
        # 中间融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 + 1, hidden_dim),  # 图像特征512维 + 表格特征1维
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        # 冻结特征提取器
        for param in self.tabular_model.parameters():
            param.requires_grad = False
        for param in self.image_model.parameters():
            param.requires_grad = False
    
    def forward(self, x_img, x_tab):
        # 通过两个模型获取特征
        tab_out = self.tabular_model(x_tab)  # [B, 1]
        
        # 获取图像特征
        img_feat = self.image_model.conv1(x_img)
        img_feat = self.image_model.bn1(img_feat)
        img_feat = self.image_model.relu(img_feat)
        img_feat = self.image_model.maxpool(img_feat)
        
        img_feat = self.image_model.layer1(img_feat)
        img_feat = self.image_model.layer2(img_feat)
        img_feat = self.image_model.layer3(img_feat)
        img_feat = self.image_model.layer4(img_feat)
        img_feat = self.image_model.avgpool(img_feat)
        img_feat = torch.flatten(img_feat, 1)  # [B, 512]
        
        # 拼接特征
        combined = torch.cat([img_feat, tab_out], dim=1)
        
        # 融合
        fused = self.fusion_fc(combined)
        out = self.fc(fused)
        return out