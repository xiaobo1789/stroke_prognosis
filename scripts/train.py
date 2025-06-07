import argparse
import json
import torch
from torch.utils.data import DataLoader
from data.dataset import StrokeDataset
from data.augmentation import CTPerfusionAugmentor
from training.trainer import StrokeTrainer
from configs.base_config import BaseConfig
from models import create_model
import sys
import os
from models import get_multimodal_model
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