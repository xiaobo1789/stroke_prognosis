class BaseConfig:
    # 数据配置
    tabular_features = [
        'age', 'gender', 'wake_up', 'arterial_fibrillation', 'referred',
        'nihss_score', 'time_to_hospital', 'ivt', 'iat', 'door_to_needle',
        'door_to_groin', 'occlusion_region', 'core_volume', 'core_percentage',
        'penumbra_volume', 'penumbra_percentage'
    ]
    image_shape = (6, 128, 128, 16)  # 6个特征图，128x128x16
    
    # 训练配置
    batch_size = 8
    epochs = 50
    learning_rate = 0.001
    weight_decay = 0.0001
    scheduler_step = 30
    scheduler_gamma = 0.1
    optimizer = 'AdamW'
    
    # 模型特定配置
    def model_specific_config(self, model_type):
        if model_type == 'TabNet':
            return {
                'n_steps': 3,
                'output_dim': 8,
                'n_shared': 2,
                'n_independent': 5
            }
        elif model_type == 'ResNet':
            return {
                'base_model': 'resnet10',
                'pretrained': False
            }
        elif model_type == 'DAFT':
            return {
                'base_model': 'resnet10',
                'bottleneck_factor': 5,
                'dropout': 0.3
            }
        else:
            return {}