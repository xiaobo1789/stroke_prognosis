# 训练工具函数
import torch

def save_checkpoint(model, epoch, path):
    torch.save(model.state_dict(), path)
