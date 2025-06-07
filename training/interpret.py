import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_tabnet_importance(model, feature_names, sample=None):
    """
    ���ӻ�TabNet������Ҫ��
    model: ѵ���õ�TabNetģ��
    feature_names: ���������б�
    sample: �ض�����������
    """
    model.eval()
    
    # ȫ��������Ҫ��
    global_importance = model.encoder.feature_importances()
    
    # ����ȫ����Ҫ��
    plt.figure(figsize=(12, 6))
    plt.barh(feature_names, global_importance)
    plt.title("Global Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("results/global_importance.png")
    
    # �ض������ľֲ���Ҫ��
    if sample is not None:
        sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        _, masks = model.encoder(sample)
        step_importances = [mask.detach().numpy().squeeze() for mask in masks]
        
        plt.figure(figsize=(12, 8))
        for i, importance in enumerate(step_importances):
            plt.subplot(len(step_importances), 1, i+1)
            plt.barh(feature_names, importance)
            plt.title(f"Step {i+1} Feature Importance")
        plt.tight_layout()
        plt.savefig("results/local_importance.png")

def visualize_daft_transformation(model, sample_img, sample_tab):
    """
    ���ӻ�DAFT�任Ч��
    model: DAFTģ��
    sample_img: ����ͼ�� [C, D, H, W]
    sample_tab: �����������
    """
    model.eval()
    
    # ��ȡ���һ���в��ǰ������
    with torch.no_grad():
        # ��ȡ����ֱ��DAFT��ǰ
        x = model.resnet.conv1(sample_img)
        x = model.resnet.bn1(x)
        x = model.resnet.relu(x)
        x = model.resnet.maxpool(x)
        x = model.resnet.layer1(x)
        x = model.resnet.layer2(x)
        x = model.resnet.layer3(x)
        
        # Ӧ��DAFTǰ������
        pre_daft = x.clone()
        
        # Ӧ��DAFT
        x = model.daft(x, sample_tab)
        
        # Ӧ��DAFT�������
        post_daft = x.clone()
    
    # ���ӻ������仯
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        # ���ѡ��һ��ͨ��
        channel_idx = np.random.randint(0, pre_daft.shape[1])
        
        # ���ѡ��һ����Ƭ
        slice_idx = np.random.randint(0, pre_daft.shape[2])
        
        # ԭʼ����
        pre_slice = pre_daft[0, channel_idx, slice_idx].cpu().numpy()
        axes[0, i].imshow(pre_slice, cmap='viridis')
        axes[0, i].set_title(f"Channel {channel_idx} (Pre-DAFT)")
        axes[0, i].axis('off')
        
        # �任������
        post_slice = post_daft[0, channel_idx, slice_idx].cpu().numpy()
        axes[1, i].imshow(post_slice, cmap='viridis')
        axes[1, i].set_title(f"Channel {channel_idx} (Post-DAFT)")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/daft_transformation.png")