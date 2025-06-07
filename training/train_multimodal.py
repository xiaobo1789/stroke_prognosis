import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.multimodal.architectures import DAFTMultimodal
from utils.data_loader import StrokeMultimodalDataset
from utils.metrics import calculate_metrics
from utils.config import load_config
from tqdm import tqdm
import numpy as np

def train_multimodal(config_path):
    # ��������
    config = load_config(config_path)
    
    # �����豸
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ��ʼ�����ݼ�
    train_dataset = StrokeMultimodalDataset(
        image_dir=config['data']['image_train_dir'],
        tabular_path=config['data']['tabular_train_path'],
        target_path=config['data']['target_train_path']
    )
    
    val_dataset = StrokeMultimodalDataset(
        image_dir=config['data']['image_val_dir'],
        tabular_path=config['data']['tabular_val_path'],
        target_path=config['data']['target_val_path']
    )
    
    # �������ݼ�����
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # ��ʼ��ģ��
    tabular_dim = len(train_dataset.tabular_features[0])
    model = DAFTMultimodal(tabular_dim=tabular_dim)
    model.to(device)
    
    # ��ʧ�������Ż���
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay']
    )
    
    # ѧϰ�ʵ�����
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # ѵ��ѭ��
    best_auc = 0.0
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        
        # ѵ���׶�
        for images, tabular, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)
            
            # ǰ�򴫲�
            outputs = model(images, tabular)
            loss = criterion(outputs, targets)
            
            # ���򴫲�
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # ��֤�׶�
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, tabular, targets in val_loader:
                images = images.to(device)
                tabular = tabular.to(device)
                targets = targets.to(device)
                
                outputs = model(images, tabular)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                
                all_preds.extend(probs[:, 1].cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # ����ָ��
        metrics = calculate_metrics(all_targets, all_preds)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
        
        # ����ѧϰ��
        scheduler.step(metrics['auc'])
        
        # �������ģ��
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            torch.save(model.state_dict(), config['training']['model_save_path'])
            print(f"Saved new best model with AUC: {best_auc:.4f}")
    
    print(f"Training complete. Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train_multimodal("configs/multimodal.yaml")