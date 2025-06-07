import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedLinearUnit(nn.Module):
    """门控线性单元"""
    def __init__(self, input_dim, output_dim):
        super(GatedLinearUnit, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return F.sigmoid(self.gate(x)) * self.fc(x)

class TabNetEncoder(nn.Module):
    """TabNet编码器"""
    def __init__(self, input_dim, output_dim=8, n_steps=3, 
                 n_shared=2, n_independent=5, virtual_batch_size=128):
        super(TabNetEncoder, self).__init__()
        self.n_steps = n_steps
        self.virtual_batch_size = virtual_batch_size
        
        # 批归一化
        self.bn = nn.BatchNorm1d(input_dim)
        
        # 特征变换层
        self.feature_transformer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim * 2),
                nn.GLU(dim=-1)
            ] for _ in range(n_steps))
        
        # 注意力变换器
        self.att_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                GatedLinearUnit(output_dim, output_dim)
            ] for _ in range(n_steps))
        
        # 输出层
        self.fc_out = nn.Linear(output_dim, 1)
    
    def forward(self, x):
        # 批归一化
        x = self.bn(x)
        
        # 初始化掩码和输出
        batch_size = x.size(0)
        masks = []
        out_agg = torch.zeros(batch_size, self.fc_out.out_features).to(x.device)
        
        # 逐步处理
        for step in range(self.n_steps):
            # 特征变换
            x_step = self.feature_transformer[step](x)
            
            # 注意力机制
            att = self.att_transformers[step](x)
            mask = F.softmax(att, dim=1)
            masks.append(mask)
            
            # 应用掩码
            x = x * mask
            
            # 聚合输出
            out_step = self.fc_out(x_step)
            out_agg += out_step
        
        # 返回最终输出和注意力掩码
        return out_agg, masks

class TabNet(nn.Module):
    """完整的TabNet模型"""
    def __init__(self, input_dim, output_dim=1, **kwargs):
        super(TabNet, self).__init__()
        self.encoder = TabNetEncoder(input_dim, **kwargs)
        self.fc = nn.Linear(1, output_dim)  # 输出层
    
    def forward(self, x):
        out, masks = self.encoder(x)
        out = self.fc(out)
        return out