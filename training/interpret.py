import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_tabnet_importance(model, feature_names, sample=None):
    """
    可视化TabNet特征重要性
    model: 训练好的TabNet模型
    feature_names: 特征名称列表
    sample: 特定样本的输入
    """
    model.eval()
    
    # 全局特征重要性
    global_importance = model.encoder.feature_importances()
    
    # 绘制全局重要性
    plt.figure(figsize=(12, 6))
    plt.barh(feature_names, global_importance)
    plt.title("Global Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("results/global_importance.png")
    
    # 特定样本的局部重要性
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
    可视化DAFT变换效果
    model: DAFT模型
    sample_img: 样本图像 [C, D, H, W]
    sample_tab: 样本表格数据
    """
    model.eval()
    
    # 获取最后一个残差块前的特征
    with torch.no_grad():
        # 提取特征直到DAFT块前
        x = model.resnet.conv1(sample_img)
        x = model.resnet.bn1(x)
        x = model.resnet.relu(x)
        x = model.resnet.maxpool(x)
        x = model.resnet.layer1(x)
        x = model.resnet.layer2(x)
        x = model.resnet.layer3(x)
        
        # 应用DAFT前的特征
        pre_daft = x.clone()
        
        # 应用DAFT
        x = model.daft(x, sample_tab)
        
        # 应用DAFT后的特征
        post_daft = x.clone()
    
    # 可视化特征变化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        # 随机选择一个通道
        channel_idx = np.random.randint(0, pre_daft.shape[1])
        
        # 随机选择一个切片
        slice_idx = np.random.randint(0, pre_daft.shape[2])
        
        # 原始特征
        pre_slice = pre_daft[0, channel_idx, slice_idx].cpu().numpy()
        axes[0, i].imshow(pre_slice, cmap='viridis')
        axes[0, i].set_title(f"Channel {channel_idx} (Pre-DAFT)")
        axes[0, i].axis('off')
        
        # 变换后特征
        post_slice = post_daft[0, channel_idx, slice_idx].cpu().numpy()
        axes[1, i].imshow(post_slice, cmap='viridis')
        axes[1, i].set_title(f"Channel {channel_idx} (Post-DAFT)")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/daft_transformation.png")