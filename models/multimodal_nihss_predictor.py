import torch
import torch.nn as nn
from models.tabnet import TabNet  # 假设TabNet在models/tabnet.py中定义
from models.resnet import ResNet  # 假设ResNet在models/resnet.py中定义
from models.resnet3d import ResNet3D  # 假设ResNet3D在models/resnet3d.py中定义

class MultimodalNIHSSPredictor(nn.Module):
    def __init__(self, clinical_features, tabnet_out=128, cnn_out=128, num_classes=4):
        super().__init__()
        
        """
        初始化多模态NIHSS预测模型。
        参数：
            clinical_features (list): 临床特征列名列表。
            tabnet_out (int): TabNet输出维度。
            cnn_out (int): 每个CNN分支输出维度。
        """
        super(MultimodalNIHSSPredictor, self).__init__()
        self.tabnet = TabNet(input_dim=len(clinical_features), output_dim=tabnet_out)
        self.cnn_ct_pre = ResNet(in_channels=1, out_channels=cnn_out)  # 治疗前CT平扫
        self.cnn_ct_post = ResNet(in_channels=1, out_channels=cnn_out)  # 治疗后CT平扫
        self.cnn_mri = ResNet3D(in_channels=1, out_channels=cnn_out)  # MRI（假设3D）
        self.fusion = nn.Linear(tabnet_out + cnn_out * 3, num_classes)  # 输出4类

    def forward(self, tabular, ct_pre, ct_post, mri):
        """
        前向传播。
        参数：
            tabular (torch.Tensor): 表格数据。
            ct_pre (torch.Tensor): 治疗前CT平扫数据。
            ct_post (torch.Tensor): 治疗后CT平扫数据。
            mri (torch.Tensor): MRI数据。
        返回：
            output (torch.Tensor): 预测的NIHSS评分。
        """
        tab_features = self.tabnet(tabular)
        ct_pre_features = self.cnn_ct_pre(ct_pre)
        ct_post_features = self.cnn_ct_post(ct_post)
        mri_features = self.cnn_mri(mri)
        fused = torch.cat([tab_features, ct_pre_features, ct_post_features, mri_features], dim=1)
        output = self.fusion(fused)
        return output