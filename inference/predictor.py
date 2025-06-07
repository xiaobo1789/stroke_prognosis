# inference/predictor.py
import torch
import numpy as np
from data.preprocessing import TabularPreprocessor, CTPerfusionPreprocessor

class StrokePredictor:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 初始化预处理器
        self.tabular_preprocessor = TabularPreprocessor()
        self.ct_preprocessor = CTPerfusionPreprocessor(
            target_shape=self.config.image_shape[1:]
        )
    
    def _load_model(self, model_path):
        # 根据路径判断模型类型
        if 'TabNet' in model_path:
            model = TabNet(input_dim=len(self.config.tabular_features))
        elif 'ResNet' in model_path:
            model = ResNet(self.config.image_shape)
        elif 'DAFT' in model_path:
            base_resnet = ResNet(self.config.image_shape)
            model = DAFTResNet(
                base_resnet, 
                tabular_dim=len(self.config.tabular_features),
                **self.config.model_specific_config('DAFT')
            )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def preprocess_input(self, tabular_data, ct_scan):
        """预处理输入数据"""
        # 表格数据
        tabular_processed = self.tabular_preprocessor.transform(
            np.array([tabular_data])
        )
        
        # CT扫描
        ct_processed = self.ct_preprocessor.process(ct_scan)
        ct_processed = np.expand_dims(ct_processed, axis=0)  # 添加批次维度
        
        return tabular_processed, ct_processed
    
    def predict(self, tabular_data, ct_scan):
        """预测中风预后结果"""
        # 预处理
        tabular, ct = self.preprocess_input(tabular_data, ct_scan)
        
        # 转换为张量
        tabular_tensor = torch.tensor(tabular, dtype=torch.float32).to(self.device)
        ct_tensor = torch.tensor(ct, dtype=torch.float32).to(self.device)
        
        # 预测
        with torch.no_grad():
            if isinstance(self.model, DAFTResNet):
                output = self.model(ct_tensor, tabular_tensor)
            elif isinstance(self.model, ResNet):
                output = self.model(ct_tensor)
            else:  # TabNet
                output = self.model(tabular_tensor)
            
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.5 else 0
        
        return {
            'prediction': prediction,
            'probability': probability,
            'interpretation': self._interpret_result(probability)
        }
    
    def _interpret_result(self, probability):
        """解释预测结果"""
        if probability > 0.7:
            return "高概率良好预后 (mRS 0-2)"
        elif probability > 0.5:
            return "中等概率良好预后"
        elif probability > 0.3:
            return "中等概率不良预后"
        else:
            return "高概率不良预后 (mRS 3-6)"