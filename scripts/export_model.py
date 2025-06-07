import torch
from models import create_model
from configs.base_config import BaseConfig

def export_model(model_type, checkpoint_path, export_dir):
    # 加载配置
    config = BaseConfig()
    
    # 创建模型
    model = create_model(model_type, config)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 导出为TorchScript
    if model_type in ['TabNet', 'ResNet']:
        # 单模态模型
        if model_type == 'TabNet':
            example_input = torch.randn(1, len(config.tabular_features))
            scripted_model = torch.jit.trace(model, example_input)
        else:
            example_input = torch.randn(1, 6, 128, 128, 16)
            scripted_model = torch.jit.trace(model, example_input)
    else:
        # 多模态模型
        example_img = torch.randn(1, 6, 128, 128, 16)
        example_tab = torch.randn(1, len(config.tabular_features))
        scripted_model = torch.jit.trace(model, (example_img, example_tab))
    
    # 保存模型
    scripted_model.save(f"{export_dir}/{model_type}_model.pt")
    print(f"Exported {model_type} model to {export_dir}/{model_type}_model.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, 
                        choices=['TabNet', 'ResNet', 'DAFT', 'LateFusion', 'HybridFusion'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--export_dir', type=str, default='exported_models',
                        help='Directory to save exported model')
    args = parser.parse_args()
    
    export_model(args.model, args.checkpoint, args.export_dir)