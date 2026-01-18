import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file = "./merged_results/1hour_1.xlsx"
df = pd.read_excel(file)

# 检查数据
print("原始数据形状:", df.shape)
print("前几行数据:")
print(df.head())

# 方法1: 如果数据中第一行确实是列名（字符串），需要跳过
# 检查第一行是否包含字符串
if df.iloc[0, 0] is not None and isinstance(df.iloc[0, 0], str):
    print("检测到第一行可能是列名，跳过第一行...")
    # 跳过第一行（列名行）
    df_clean = df.iloc[1:].reset_index(drop=True)

    # 确保数据类型正确
    for i in range(df_clean.shape[1]):
        df_clean.iloc[:, i] = pd.to_numeric(df_clean.iloc[:, i], errors='coerce')

    # 删除可能存在的NaN值
    df_clean = df_clean.dropna()

    print("清洗后数据形状:", df_clean.shape)
    print("清洗后数据前几行:")
    print(df_clean.head())

    # 将清洗后的数据赋值回df
    df = df_clean

# 方法2: 如果数据已经是纯数值，但第一行是第一维数据
else:
    print("数据已经是数值类型，直接使用...")

    # 确保数据类型正确
    for i in range(df.shape[1]):
        df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], errors='coerce')

    # 删除可能存在的NaN值
    df = df.dropna()
    print("处理后数据形状:", df.shape)


# 数据预处理函数
def prepare_data(data, sequence_length = 10):
    """
    准备LSTM输入数据
    data: 输入数据，形状为 (n_samples, n_features)
    sequence_length: 序列长度
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # 序列数据
        y.append(data[i + sequence_length])  # 下一个时间点的数据
    return np.array(X), np.array(y)


# 提取特征数据
# 第一维：时间数据（通常用于索引或可视化）
time_data = df.iloc[:, 0].values.astype(float)

# 第二到第五维：四维特征数据
features = df.iloc[:, 1:5].values.astype(float)

print(f"时间数据形状: {time_data.shape}")
print(f"特征数据形状: {features.shape}")
print(f"特征数据前5行:\n{features[:5]}")
print(f"特征数据统计:\n均值: {features.mean(axis=0)}, 标准差: {features.std(axis=0)}")

# 创建特征名称（用于可视化）
feature_names = [f'Feature_{i + 1}' for i in range(features.shape[1])]

# 可视化原始数据特征
plt.figure(figsize=(15, 10))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(features[:200, i], label=feature_names[i])  # 只显示前200个点以便查看
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'原始数据 - {feature_names[i]}')
    plt.grid(True, alpha=0.3)
    plt.legend()
plt.suptitle('四维时间序列数据可视化', fontsize=16)
plt.tight_layout()
plt.show()

# 标准化数据
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)

print(f"标准化后特征形状: {features_scaled.shape}")
print(f"标准化后前5行:\n{features_scaled[:5]}")

# 设置序列长度
sequence_length = 168  # 根据数据特性调整

# 检查数据长度是否足够
if len(features_scaled) <= sequence_length:
    print(f"警告: 数据长度({len(features_scaled)})小于序列长度({sequence_length})")
    # 自动调整序列长度
    sequence_length = max(5, len(features_scaled) // 10)
    print(f"自动调整序列长度为: {sequence_length}")

# 准备数据集
X, y = prepare_data(features_scaled, sequence_length)

print(f"X (输入序列) 形状: {X.shape}")
print(f"y (目标值) 形状: {y.shape}")

# 划分训练集和测试集
# 对于时间序列，通常按时间顺序划分
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

print(f"训练张量形状: {X_train_tensor.shape}")
print(f"测试张量形状: {X_test_tensor.shape}")

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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


# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = LSTMModel(
    input_size=4,  # 4维输入数据
    hidden_size=256,  # LSTM隐藏层大小
    num_layers=4,  # LSTM层数
    output_size=4  # 预测4维输出
).to(device)

# 打印模型结构
print("模型结构:")
print(model)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)


# 训练模型
def train_model(model, train_loader, test_loader, epochs=50):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)

        # 学习率调度
        scheduler.step(avg_test_loss)

        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1:03d}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Test Loss: {avg_test_loss:.6f}')

    return train_losses, test_losses


# 开始训练
print("\n开始训练模型...")
epochs = 75
train_losses, test_losses = train_model(model, train_loader, test_loader, epochs)

# 绘制训练损失
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(test_losses, label='Test Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 预测并可视化结果
def plot_predictions(model, X_test, y_test, scaler, time_test=None, feature_names=None):
    if feature_names is None:
        feature_names = [f'Feature_{i + 1}' for i in range(4)]

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_tensor).cpu().numpy()

    # 反标准化
    predictions_original = scaler.inverse_transform(predictions)
    y_test_original = scaler.inverse_transform(y_test)

    # 创建时间索引
    if time_test is None:
        time_test = np.arange(len(y_test_original))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        ax.plot(time_test[:100], y_test_original[:100, i],
                label='Actual', alpha=0.7, linewidth=2, marker='o', markersize=3)
        ax.plot(time_test[:100], predictions_original[:100, i],
                label='Predicted', alpha=0.7, linestyle='--', linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f'{feature_names[i]} - Actual vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('LSTM预测结果对比', fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig("1hour_lstm_predict.png")

    return predictions_original, y_test_original


# 获取测试集对应的时间
time_test = time_data[split_idx + sequence_length:split_idx + sequence_length + len(y_test)]

# 绘制预测结果
print("\n绘制预测结果...")
predictions, actuals = plot_predictions(
    model, X_test, y_test, scaler,
    time_test=time_test,
    feature_names=feature_names
)

# 计算评估指标
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(predictions, actuals, feature_names):
    metrics = {}
    print("\n模型性能评估:")
    print("=" * 60)

    for i in range(len(feature_names)):
        mae = mean_absolute_error(actuals[:, i], predictions[:, i])
        mse = mean_squared_error(actuals[:, i], predictions[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals[:, i], predictions[:, i])

        # 计算相对误差
        mean_actual = np.mean(np.abs(actuals[:, i]))
        relative_error = (rmse / mean_actual * 100) if mean_actual != 0 else np.nan

        metrics[feature_names[i]] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'Relative_Error(%)': relative_error
        }

        print(f"\n{feature_names[i]}:")
        print(f"  MAE: {mae:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²: {r2:.4f}")
        print(f"  相对误差: {relative_error:.2f}%")

    # 总体指标
    overall_mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
    overall_rmse = np.sqrt(mean_squared_error(actuals.flatten(), predictions.flatten()))

    metrics['Overall'] = {
        'MAE': overall_mae,
        'RMSE': overall_rmse
    }

    print("\n" + "=" * 60)
    print(f"总体指标:")
    print(f"  MAE: {overall_mae:.6f}")
    print(f"  RMSE: {overall_rmse:.6f}")

    return metrics


# 计算并显示评估指标
metrics = calculate_metrics(predictions, actuals, feature_names)

# 保存完整模型信息
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'sequence_length': sequence_length,
    'feature_names': feature_names,
    'input_size': 4,
    'hidden_size': 128,
    'num_layers': 2,
    'output_size': 4
}, 'lstm_time_series_model_complete.pth')

print("\n模型已保存为 'lstm_time_series_model_complete.pth'")


# 预测未来的函数
def predict_future(model, last_sequence, steps=10, scaler=None, sequence_length=20):
    """
    预测未来多个时间步
    last_sequence: 最后sequence_length个数据点
    steps: 要预测的未来步数
    """
    model.eval()
    predictions = []

    # 初始输入序列
    current_sequence = last_sequence.copy()

    with torch.no_grad():
        for step in range(steps):
            # 准备当前序列
            sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)

            # 预测下一个点
            pred = model(sequence_tensor).cpu().numpy()
            predictions.append(pred[0])

            # 更新序列：移除第一个点，加入预测值
            current_sequence = np.vstack([current_sequence[1:], pred])

    # 反标准化
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)

    return np.array(predictions)


# 示例：预测未来10个时间步
print("\n预测未来10个时间步...")
# 使用最后sequence_length个数据点作为起始
last_sequence_scaled = features_scaled[-sequence_length:]
future_predictions = predict_future(
    model, last_sequence_scaled,
    steps=10, scaler=scaler, sequence_length=sequence_length
)

print(f"未来10个时间步的预测值:")
for i, pred in enumerate(future_predictions):
    print(f"  时间步 {i + 1}: {pred}")

# 可视化未来预测
plt.figure(figsize=(15, 10))
for i in range(4):
    plt.subplot(2, 2, i + 1)

    # 历史数据（最后50个点）
    history = features[-50:, i]
    plt.plot(range(len(history)), history, 'b-', label='Historical', alpha=0.7)

    # 未来预测
    future = future_predictions[:, i]
    plt.plot(range(len(history), len(history) + len(future)),
             future, 'r--', label='Forecast', alpha=0.7, marker='o')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'{feature_names[i]} - Historical and Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.suptitle('未来10个时间步的预测', fontsize=16)
plt.tight_layout()
plt.show()
plt.savefig("1hour_sequence_predict.png")