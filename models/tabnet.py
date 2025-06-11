import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedLinearUnit(nn.Module):
    """门控线性单元（与nn.GLU功能一致，可选保留）"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return F.sigmoid(self.gate(x)) * self.fc(x)

class VirtualBatchNorm(nn.Module):
    """虚拟批次归一化（适用于小批量训练）"""
    def __init__(self, num_features, virtual_batch_size=128):
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(num_features, affine=False)  # 无参数版本
        self.gamma = nn.Parameter(torch.ones(num_features))   # 可学习的缩放
        self.beta = nn.Parameter(torch.zeros(num_features))   # 可学习的偏移

    def forward(self, x):
        batch_size = x.size(0)
        if batch_size <= self.virtual_batch_size:
            return self.bn(x) * self.gamma + self.beta
        
        # 分块计算全局统计量
        num_blocks = (batch_size + self.virtual_batch_size - 1) // self.virtual_batch_size
        blocks = torch.split(x, self.virtual_batch_size, dim=0)
        
        # 计算全局均值和方差
        mean = sum(block.mean(dim=0) for block in blocks) / num_blocks
        var = sum(block.var(dim=0, unbiased=False) for block in blocks) / num_blocks
        
        # 归一化每个块
        normalized_blocks = [(block - mean) / (var.sqrt() + 1e-5) * self.gamma + self.beta 
                            for block in blocks]
        return torch.cat(normalized_blocks, dim=0)

class TabNetEncoder(nn.Module):
    """改进版TabNet编码器（支持共享/独立特征变换）"""
    def __init__(self, input_dim, output_dim=8, n_steps=3, 
                 n_shared=2, n_independent=2, virtual_batch_size=128):
        super().__init__()
        self.n_steps = n_steps
        self.output_dim = output_dim
        self.virtual_batch_size = virtual_batch_size

        # 输入归一化
        self.input_bn = VirtualBatchNorm(input_dim, virtual_batch_size)

        # 共享特征变换层（所有step共享）
        self.shared_layers = nn.ModuleList()
        for i in range(n_shared):
            in_dim = input_dim if i == 0 else output_dim
            self.shared_layers.append(nn.Sequential(
                nn.Linear(in_dim, output_dim * 2),
                nn.GLU(dim=-1),
                VirtualBatchNorm(output_dim, virtual_batch_size)
            ))

        # 独立特征变换层（每个step独立）
        self.step_layers = nn.ModuleList()
        for _ in range(n_steps):
            layers = nn.ModuleList()
            for i in range(n_independent):
                in_dim = output_dim if i == 0 else output_dim
                layers.append(nn.Sequential(
                    nn.Linear(in_dim, output_dim * 2),
                    nn.GLU(dim=-1),
                    VirtualBatchNorm(output_dim, virtual_batch_size)
                ))
            self.step_layers.append(layers)

        # 注意力变换层（使用前一步输出生成掩码）
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, input_dim),  # 映射回输入维度生成掩码
                VirtualBatchNorm(input_dim, virtual_batch_size),
                nn.Softmax(dim=-1)
            ) for _ in range(n_steps)
        ])

    def forward(self, x):
        x = self.input_bn(x)
        batch_size = x.size(0)
        masks = []
        prior = torch.ones_like(x)  # 初始先验掩码（控制特征选择的稀疏性）
        output_agg = torch.zeros(batch_size, self.output_dim).to(x.device)

        for step in range(self.n_steps):
            # 特征变换：共享层 + 独立层
            ft = x
            for layer in self.shared_layers:
                ft = layer(ft)
            for layer in self.step_layers[step]:
                ft = layer(ft)
            
            # 生成注意力掩码（使用前一步输出和先验）
            if step > 0:
                attention_input = output_agg
            else:
                attention_input = ft  # 第一步使用初始特征
            mask = self.attention_layers[step](attention_input * prior)
            masks.append(mask)
            
            # 应用掩码并更新先验（鼓励稀疏选择）
            x = x * mask
            prior = prior * (1 - mask)  # 先验衰减
            
            # 聚合决策输出（论文中的决策路径）
            output_agg += F.relu(ft)  # 使用ReLU激活聚合

        return output_agg, masks

class TabNet(nn.Module):
    """完整TabNet模型（支持分类/回归）"""
    def __init__(self, input_dim, output_dim=1, is_classification=True, **kwargs):
        super().__init__()
        self.encoder = TabNetEncoder(input_dim, **kwargs)
        self.head = nn.Linear(kwargs.get('output_dim', 8), output_dim)
        self.is_classification = is_classification

    def forward(self, x):
        features, masks = self.encoder(x)
        logits = self.head(features)
        
        if self.is_classification and self.head.out_features == 1:
            return torch.sigmoid(logits), masks  # 二分类
        elif self.is_classification:
            return F.softmax(logits, dim=-1), masks  # 多分类
        return logits, masks  # 回归任务