# core/lprnet_model.py
import torch.nn as nn
import torch

class LPRNet(nn.Module):
    """
    一个简化的 LPRNet 模型结构。
    确保这个结构和你训练时用的 *完全一致*。
    """
    def __init__(self, num_classes):
        super(LPRNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # 94x24
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)), # 92x22
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 92x22
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)), # 90x20
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1), # 45x20
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 45x20
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)), # 22x18
        )
        
        # 展平特征，为 RNN 准备
        self.rnn = nn.LSTM(256 * 18, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes) # 512 = 256 * 2 (bidirectional)

    def forward(self, x):
        # x: (batch_size, 3, 24, 94)
        x = self.backbone(x)
        
        # 调整维度以匹配 RNN 输入
        # x: (b, 256, 18, 22) -> (b, 22, 256*18)
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2) # (b, w, c, h)
        x = x.reshape(b, w, c * h) # (b, 22, 256*18)
        
        x, _ = self.rnn(x) # (b, 22, 512)
        
        x = self.fc(x) # (b, 22, num_classes)
        
        # CTC Loss 需要 (seq_len, batch, num_classes)
        x = x.permute(1, 0, 2) # (22, b, num_classes)
        
        return x