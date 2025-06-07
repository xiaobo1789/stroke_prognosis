import numpy as np
import pandas as pd
# 关键修正：先启用实验性功能（必须在导入 IterativeImputer 之前）
from sklearn.experimental import enable_iterative_imputer
# 再导入 IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom

class TabularPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = IterativeImputer(random_state=42)
        
    def fit_transform(self, data):
        # 处理缺失值
        imputed_data = self.imputer.fit_transform(data)
        
        # 标准化
        scaled_data = self.scaler.fit_transform(imputed_data)
        return scaled_data
    
    def transform(self, data):
        imputed_data = self.imputer.transform(data)
        return self.scaler.transform(imputed_data)

class CTPerfusionPreprocessor:
    def __init__(self, target_shape=(128, 128, 16)):
        self.target_shape = target_shape
        
    def process(self, ct_scan):
        """
        处理CT灌注扫描数据
        """
        # 1. 标准化尺寸
        if ct_scan.shape != self.target_shape:
            factors = [
                self.target_shape[0] / ct_scan.shape[0],
                self.target_shape[1] / ct_scan.shape[1],
                self.target_shape[2] / ct_scan.shape[2]
            ]
            ct_scan = zoom(ct_scan, factors, order=1)
        
        # 2. 生成特征图
        # 原始特征图：平均和求和
        averaged_map = np.mean(ct_scan, axis=-1)
        summed_map = np.sum(ct_scan, axis=-1)
        
        # 计算参数图 (假设已通过专业软件预处理)
        # 实际应用中应从DICOM文件计算或从处理软件获取
        cbv_map = self.calculate_cbv(ct_scan)
        cbf_map = self.calculate_cbf(ct_scan)
        mtt_map = self.calculate_mtt(ct_scan)
        ttp_map = self.calculate_ttp(ct_scan)
        
        # 组合所有特征图 (6个通道)
        feature_maps = np.stack([
            averaged_map, summed_map, 
            cbv_map, cbf_map, mtt_map, ttp_map
        ], axis=-1)
        
        return feature_maps
    
    def calculate_cbv(self, ct_scan, baseline_intensity):
        """基于时间-密度曲线计算脑血容量（CBV）"""
        # 真实实现需：1. 减去基线强度；2. 时间积分；3. 校正因子
        contrast = ct_scan - baseline_intensity
        return np.trapz(contrast, axis=-1)  # 示例积分
    
    def calculate_cbf(self, ct_scan):
        # 简化的CBF计算
        return np.max(ct_scan, axis=-1) - np.min(ct_scan, axis=-1)
    
    def calculate_mtt(self, ct_scan):
        # 简化的MTT计算
        return np.argmax(ct_scan, axis=-1)
    
    def calculate_ttp(self, ct_scan):
        # 简化的TTP计算
        return np.argmin(ct_scan, axis=-1)