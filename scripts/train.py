import argparse
import json
import torch
from configs.base_config import required_clinical_features
from scripts.modified_stroke_dataset_loader import StrokeDataset
from models.multimodal_nihss_predictor import MultimodalNIHSSPredictor
from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataset import StrokeDataset
from data.augmentation import CTPerfusionAugmentor
from training.trainer import StrokeTrainer
from configs.base_config import BaseConfig
from models import create_model
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from models import get_multimodal_model
def train_model(model, dataloader, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 加权交叉熵损失
    class_counts = [100, 50, 30, 20]  # 假设的类别样本数
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        for batch in dataloader:
            ct_pre, ct_post, mr_post, tabular, labels = batch.values()
            ct_pre, ct_post, mr_post, tabular, labels = (
                ct_pre.to(device), ct_post.to(device), mr_post.to(device),
                tabular.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            outputs = model(ct_pre, ct_post, mr_post, tabular)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 示例数据加载器（需根据实际数据集实现）
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# train_model(model, dataloader)
# 创建数据集
train_dataset = StrokeDataset(csv_file='data/train.csv', root_dir='data', clinical_features=required_clinical_features)
criterion = nn.MSELoss()    
# 创建模型
model = MultimodalNIHSSPredictor(clinical_features=required_clinical_features)
# 获取项目根目录（即当前脚本的父目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)  # 添加到Python路径
class_weights = 1.0 / torch.tensor([count0, count1, count2, count3], dtype=torch.float)
criterion = CrossEntropyLoss(weight=class_weights)
def main(model_type):
    # 加载配置
    config = BaseConfig()
    model_config = config.model_specific_config(model_type)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据增强
    augmentor = CTPerfusionAugmentor(apply_augmentation=True)
    
    # 加载数据集
    train_dataset = StrokeDataset(
        'data/train.csv', 
        'data/ct_scans/train',
        is_train=True
    )
    val_dataset = StrokeDataset(
        'data/val.csv', 
        'data/ct_scans/val',
        is_train=False
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 初始化模型
    model = create_model(model_type, config)
    print(f"Created {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 训练器
    trainer = StrokeTrainer(
        model, 
        train_loader, 
        val_loader, 
        device, 
        config,
        augmentor=augmentor
    )
    
    # 训练模型
    history = trainer.train(config.epochs)
    
    # 保存训练历史
    with open(f"results/history_{model_type}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training completed for {model_type}. Results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, 
                        choices=['TabNet', 'ResNet', 'DAFT', 'LateFusion', 'HybridFusion'],
                        help='Model type to train')
    args = parser.parse_args()
    
    main(args.model)