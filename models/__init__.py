from .tabnet import TabNet
from .resnet import ResNet10
from .daft import DAFT, DAFTResNet
from .fusion_models import LateFusion, HybridFusion
from .multimodal_daft import MultimodalDAFT, get_multimodal_model
def create_model(model_type, config):
    """根据配置创建模型"""
    if model_type == 'TabNet':
        return TabNet(
            input_dim=len(config.tabular_features),
            output_dim=1,
            n_steps=config.n_steps,
            n_shared=config.n_shared,
            n_independent=config.n_independent
        )
    
    elif model_type == 'ResNet':
        return ResNet10(
            in_channels=6,  # 6个特征图
            num_classes=1
        )
    
    elif model_type == 'DAFT':
        base_resnet = ResNet10(in_channels=6)
        return DAFTResNet(
            base_resnet,
            tabular_dim=len(config.tabular_features),
            num_classes=1,
            bottleneck_factor=config.bottleneck_factor
        )
    
    elif model_type == 'LateFusion':
        tabular_model = TabNet(
            input_dim=len(config.tabular_features),
            output_dim=1
        )
        image_model = ResNet10(in_channels=6)
        return LateFusion(tabular_model, image_model)
    
    elif model_type == 'HybridFusion':
        tabular_model = TabNet(
            input_dim=len(config.tabular_features),
            output_dim=1
        )
        image_model = ResNet10(in_channels=6)
        return HybridFusion(tabular_model, image_model)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")