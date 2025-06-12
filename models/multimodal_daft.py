import torch
import torch.nn as nn
from models.tabnet import TabNet
from models.resnet3d import generate_resnet10_3d

class DAFTBlock(nn.Module):
    
    def __init__(self, img_channels, tabular_dim, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)  # ȫ��ƽ���ػ�
        self.fc = nn.Sequential(
            nn.Linear(img_channels + tabular_dim, (img_channels + tabular_dim) // reduction),
            nn.ReLU(inplace=True),
            nn.Linear((img_channels + tabular_dim) // reduction, img_channels * 2),
            nn.Dropout(0.3)  # ���dropout��ֹ�����
        )
        
    def forward(self, img_feats, tabular):
        # ����: img_feats [B, C, D, H, W], tabular [B, T]
        pooled = self.gap(img_feats).flatten(1)  # [B, C]
        combined = torch.cat([pooled, tabular], dim=1)  # [B, C + T]
        params = self.fc(combined)  # [B, 2*C]
        
        # ����scale��shift����
        scale, shift = params.chunk(2, dim=1)  # ��Ϊ[B, C]
        
        # ��չά��ƥ������ͼ [B, C, 1, 1, 1]
        scale = scale.view(*scale.shape, 1, 1, 1)
        shift = shift.view(*shift.shape, 1, 1, 1)
        
        # Ӧ�÷���任
        return img_feats * (scale + 1) + shift  # ������ʹ�����Լ���

class MultimodalDAFT(nn.Module):
    
    def __init__(self, tabular_input_dim, num_classes=4, use_tabnet=True):
        super().__init__()
        
        # 1. ������ݷ�֧
        self.use_tabnet = use_tabnet
        if use_tabnet:
            self.tabular_branch = TabNet(
                input_dim=tabular_input_dim,
                output_dim=128,  # TabNet�������ά��
                n_d=64,
                n_a=64,
                n_steps=3
            )
        else:
            self.tabular_branch = nn.Sequential(
                nn.Linear(tabular_input_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True)
            )
        
        # 2. Ӱ�����ݷ�֧
        # ����ǰCT
        self.ct_pre_branch = generate_resnet10_3d(
            in_channels=1,  # ��ͨ��CT
            base_filters=32,  # ���ٲ���
            num_classes=128   # �������ά��
        )
        
        # ���ƺ�CT
        self.ct_post_branch = generate_resnet10_3d(
            in_channels=1,
            base_filters=32,
            num_classes=128
        )
        
        # ���ƺ�MRI
        self.mr_post_branch = generate_resnet10_3d(
            in_channels=1,
            base_filters=32,
            num_classes=128
        )
        
        # 3. DAFT�ں�ģ�� (ÿ��Ӱ���֧һ��)
        self.daft_ct_pre = DAFTBlock(
            img_channels=128, 
            tabular_dim=128,
            reduction=4
        )
        self.daft_ct_post = DAFTBlock(
            img_channels=128, 
            tabular_dim=128,
            reduction=4
        )
        self.daft_mr_post = DAFTBlock(
            img_channels=128, 
            tabular_dim=128,
            reduction=4
        )
        
        # 4. �����ںϷ�����
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # ȫ��ƽ���ػ�
            nn.Flatten(),
            nn.Linear(128 * 3, 256),  # 3��Ӱ���֧
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, ct_pre, ct_post, mr_post, tabular):
        # 1. ����������
        tab_feats = self.tabular_branch(tabular)  # [B, 128]
        
        # 2. ����Ӱ������
        ct_pre_feats = self.ct_pre_branch(ct_pre)  # [B, 128, D, H, W]
        ct_post_feats = self.ct_post_branch(ct_post)
        mr_post_feats = self.mr_post_branch(mr_post)
        
        # 3. Ӧ��DAFT�ں�
        fused_ct_pre = self.daft_ct_pre(ct_pre_feats, tab_feats)
        fused_ct_post = self.daft_ct_post(ct_post_feats, tab_feats)
        fused_mr_post = self.daft_mr_post(mr_post_feats, tab_feats)
        
        # 4. ƴ���ںϺ������
        combined_feats = torch.cat([
            fused_ct_pre, 
            fused_ct_post, 
            fused_mr_post
        ], dim=1)  # [B, 128*3, D, H, W]
        
        # 5. ����
        return self.classifier(combined_feats)

# ������������ȡģ��
def get_multimodal_model(tabular_dim, num_classes=4, use_pretrained=False):
    model = MultimodalDAFT(
        tabular_input_dim=tabular_dim,
        num_classes=num_classes,
        use_tabnet=True
    )
    
    # ����Ԥѵ��Ȩ�� (�����)
    if use_pretrained:
        try:
            model.load_state_dict(torch.load('pretrained/multimodal_daft.pth'))
            print("Loaded pretrained weights")
        except:
            print("Pretrained weights not found, using random initialization")
    
    return model