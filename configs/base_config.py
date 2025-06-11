class BaseConfig:
    # 数据配置
    tabular_features = [
        's', 'treatment', 'name', 'id', 'Xid', 'date',
        'CT1', 'CT2', 'CT3', 'MR1', 'MR2', 'MR-TOF',
        'nihss', 'baseline1', 'n7', 'time', 'time1', 'time2', 'time3',
        'dnt', 'hypertens', 'diabetes', 'hyperlipi', 'atrial', 'coronary',
        'heart', 'previous', 'alcohol', 'smoking', 'baseline2', 'sbp1', 'sbp2',
        'sbp3', 'sub4', 'baseline3', 'dbp1', 'dbp2', 'dbp3', 'dbp4',
        'occlusion', 'fibrinoge', 'ddimer', 'fibrinoge1', 'k', 'na', 'cl', 'ca',
        'alanine', 'aspartate', 'alkaline', 'glutamyl', 'total', 'field1',
        'total1', 'direct', 'urea', 'creatinin', 'ua', 'total2', 'glucose', 'd',
        'leucocyte', 'neutrophi', 'percentag', 'percentag1', 'percentag2',
        'percentag3', 'neutrophi1', 'eosinophi', 'basophils', 'monocytes',
        'lymphocyt', 'hemoglobi', 'red', 'blood', 'creactiv'
    ]
    
    required_clinical_features = [
        'nihss',  # 神经功能评分
        'hypertens', 'diabetes', 'hyperlipi', 'atrial',  # 基础疾病（高血压、糖尿病、高血脂、房颤）
        'sbp1', 'sbp2', 'sbp3', 'dbp1', 'dbp2', 'dbp3',  # 血压数据
        'dnt', 'time',  # 时间维度特征（类似示例中的 time_to_hospital 等）
        # 若有其他核心临床指标（如年龄、性别，需数据中存在对应字段），可在此补充
    ]
    # 训练配置
    image_shape = (6, 128, 128, 16)  # 6个特征图，128x128x16
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