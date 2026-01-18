import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_layers=4, output_size=4):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=False  # 对于时间序列预测，单向LSTM通常足够
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # 初始化隐藏状态
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        last_time_step = lstm_out[:, -1, :]

        # 全连接层
        output = self.fc(last_time_step)
        return output


def load_saved_model(model_path):
    """加载保存的完整模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载保存的checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 重新创建模型
    model = LSTMModel(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        output_size=checkpoint['output_size']
    ).to(device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式

    # 加载其他必要信息
    scaler = checkpoint['scaler']
    sequence_length = checkpoint['sequence_length']
    feature_names = checkpoint.get('feature_names', [f'Feature_{i + 1}' for i in range(4)])

    return {
        'model': model,
        'scaler': scaler,
        'sequence_length': sequence_length,
        'feature_names': feature_names,
        'device': device
    }


# 使用示例
model_info = load_saved_model('lstm_time_series_model_complete.pth')

model = model_info['model']
scaler = model_info['scaler']
sequence_length = model_info['sequence_length']
device = model_info['device']

