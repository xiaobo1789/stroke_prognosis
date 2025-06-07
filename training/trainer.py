# 在原有基础上增加数据增强支持
class StrokeTrainer:
    def __init__(self, model, train_loader, val_loader, device, config, augmentor=None):
        # ... 原有代码 ...
        self.augmentor = augmentor
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # 获取数据
            tabular = batch['tabular'].to(self.device)
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device).float()
            
            # 应用数据增强 (仅对图像)
            if self.augmentor:
                images = self.augmentor(images)
            
            # ... 其余代码保持不变 ...
    
    # ... 其余代码保持不变 ...