import torch
import torch.nn as nn
from models.tabnet import TabNet  # ����TabNet��models/tabnet.py�ж���
from models.resnet import ResNet  # ����ResNet��models/resnet.py�ж���
from models.resnet3d import ResNet3D  # ����ResNet3D��models/resnet3d.py�ж���

class MultimodalNIHSSPredictor(nn.Module):
    def __init__(self, clinical_features, tabnet_out=128, cnn_out=128, num_classes=4):
        super().__init__()
        
        """
        ��ʼ����ģ̬NIHSSԤ��ģ�͡�
        ������
            clinical_features (list): �ٴ����������б�
            tabnet_out (int): TabNet���ά�ȡ�
            cnn_out (int): ÿ��CNN��֧���ά�ȡ�
        """
        super(MultimodalNIHSSPredictor, self).__init__()
        self.tabnet = TabNet(input_dim=len(clinical_features), output_dim=tabnet_out)
        self.cnn_ct_pre = ResNet(in_channels=1, out_channels=cnn_out)  # ����ǰCTƽɨ
        self.cnn_ct_post = ResNet(in_channels=1, out_channels=cnn_out)  # ���ƺ�CTƽɨ
        self.cnn_mri = ResNet3D(in_channels=1, out_channels=cnn_out)  # MRI������3D��
        self.fusion = nn.Linear(tabnet_out + cnn_out * 3, num_classes)  # ���4��

    def forward(self, tabular, ct_pre, ct_post, mri):
        """
        ǰ�򴫲���
        ������
            tabular (torch.Tensor): ������ݡ�
            ct_pre (torch.Tensor): ����ǰCTƽɨ���ݡ�
            ct_post (torch.Tensor): ���ƺ�CTƽɨ���ݡ�
            mri (torch.Tensor): MRI���ݡ�
        ���أ�
            output (torch.Tensor): Ԥ���NIHSS���֡�
        """
        tab_features = self.tabnet(tabular)
        ct_pre_features = self.cnn_ct_pre(ct_pre)
        ct_post_features = self.cnn_ct_post(ct_post)
        mri_features = self.cnn_mri(mri)
        fused = torch.cat([tab_features, ct_pre_features, ct_post_features, mri_features], dim=1)
        output = self.fusion(fused)
        return output