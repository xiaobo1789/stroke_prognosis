import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

# 3D Bottleneck���̳�2D Bottleneck������Ϊ3D�����
class Bottleneck3D(Bottleneck):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__(inplanes, planes, stride, downsample, groups=1, base_width=64)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=stride,
                               padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

class ResNet3D(nn.Module):
    def __init__(self, pretrained_path=None):
        super(ResNet3D, self).__init__()
        self.inplanes = 64
        # 3D����㣨��ͨ��ҽѧӰ����CT��
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # 3D ResNet-50 �в�飨ƥ��MedicalNet�ṹ��
        self.layer1 = self._make_layer(Bottleneck3D, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck3D, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck3D, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck3D, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * Bottleneck3D.expansion, 1000)  # ԭʼ����ͷ�����滻��
        
        # ����Ԥѵ��Ȩ��
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def load_pretrained(self, path):
        pretrained_dict = torch.load(path, map_location='cpu')  # ����Ȩ�أ�֧��CPU/GPU��
        model_dict = self.state_dict()
        # ���˲�ƥ��Ĳ㣨��fc����ͷ������������ȡ���֣�
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
    
    def forward(self, x):
        # x: (B, 1, D, H, W) 3DҽѧӰ������
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x  # ����2048ά���������ڶ�ģ̬�ںϣ�

# ʹ��ʾ������ѵ���ű��У�
if __name__ == "__main__":
    pretrained_path = r"F:\stroke_prognosis\MedicalNet_pytorch_files2\pretrain\resnet_50.pth"
    model = ResNet3D(pretrained_path=pretrained_path)
    print(model)  # ��ӡģ�ͽṹ����֤Ȩ�ؼ���