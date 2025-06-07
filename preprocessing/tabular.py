from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

class TabularPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()  # 添加
        self.imputer = IterativeImputer(random_state=42)  # 添加

    def fit_transform(self, data):
        data_imputed = data.copy()
        data_imputed['door_to_needle'] = data_imputed['door_to_needle'].fillna(0)
        data_imputed['door_to_groin'] = data_imputed['door_to_groin'].fillna(0)
        other_cols = [col for col in data.columns if col not in ['door_to_needle', 'door_to_groin']]
        data_imputed[other_cols] = self.imputer.fit_transform(data_imputed[other_cols])  # 使用已初始化的 imputer
        data_imputed[continuous_cols] = self.scaler.fit_transform(data_imputed[continuous_cols])  # 使用已初始化的 scaler
        return data_imputed
    def transform(self, data):
        """转换新数据"""
        data_imputed = data.copy()
        data_imputed['door_to_needle'] = data_imputed['door_to_needle'].fillna(0)
        data_imputed['door_to_groin'] = data_imputed['door_to_groin'].fillna(0)
        
        other_cols = [col for col in data.columns if col not in ['door_to_needle', 'door_to_groin']]
        data_imputed[other_cols] = self.imputer.transform(data_imputed[other_cols])
        
        continuous_cols = [
            'age', 'nihss_score', 'time_to_hospital', 
            'door_to_needle', 'door_to_groin',
            'penumbra_volume', 'penumbra_percent', 
            'core_volume', 'core_percent'
        ]
        data_imputed[continuous_cols] = self.scaler.transform(data_imputed[continuous_cols])
        
        return data_imputed