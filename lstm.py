import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')

excel_file = "./merged_results/1hour_1.xlsx"
df = pd.read_excel(excel_file, skiprows = 1)

# 1. 数据预处理函数
def prepare_data(df, time_steps=10, future_steps=1):
    """
    准备LSTM训练数据

    参数:
    df: 包含时间序列的DataFrame，第一列是时间，2-5列是四维数据
    time_steps: 使用多少时间步长的历史数据
    future_steps: 预测未来多少步长
    """
    # 提取四维特征数据（假设列名为col1, col2, col3, col4）
    # 如果没有列名，使用数字索引
    if df.shape[1] >= 5:
        features = df.iloc[:, 1:5].values  # 2-5列是四维数据
    else:
        raise ValueError("DataFrame需要至少5列：时间列 + 4个特征列")

    # 标准化数据（对于LSTM很重要）
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    # 创建时间序列样本
    X, y = [], []

    for i in range(len(features_scaled) - time_steps - future_steps + 1):
        # 历史数据窗口
        X.append(features_scaled[i:(i + time_steps), :])

        # 未来数据（多步预测）
        if future_steps == 1:
            # 单步预测
            y.append(features_scaled[i + time_steps, :])
        else:
            # 多步预测
            y.append(features_scaled[(i + time_steps):(i + time_steps + future_steps), :])

    X = np.array(X)
    y = np.array(y)

    # 如果是多步预测，需要重塑y的形状
    if future_steps > 1:
        y = y.reshape(y.shape[0], future_steps * 4)  # 4个特征

    return X, y, scaler


# 2. 创建LSTM模型
def create_lstm_model(time_steps, n_features, future_steps=1):
    """
    创建LSTM模型

    参数:
    time_steps: 输入时间步长
    n_features: 特征数量（这里是4）
    future_steps: 预测步长
    """
    model = Sequential()

    # 第一层LSTM
    model.add(Input(shape=(time_steps, n_features)))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))

    # 第二层LSTM
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    # 第三层LSTM
    model.add(LSTM(32))
    model.add(Dropout(0.2))

    # 输出层
    if future_steps == 1:
        # 单步预测：4个输出对应4个特征
        model.add(Dense(4))
    else:
        # 多步预测：future_steps * 4个输出
        model.add(Dense(future_steps * 4))

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


# 3. 完整训练和预测流程
def train_and_predict(df, time_steps=20, future_steps=1, test_size=0.2):
    """
    完整的训练和预测流程

    参数:
    df: 输入数据
    time_steps: 时间窗口大小
    future_steps: 预测未来步数
    test_size: 测试集比例
    """
    # 准备数据
    X, y, scaler = prepare_data(df, time_steps, future_steps)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")

    # 创建模型
    model = create_lstm_model(
        time_steps=time_steps,
        n_features=4,
        future_steps=future_steps
    )

    model.summary()

    # 设置早停
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # 评估模型
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试集 MSE: {test_mse:.6f}")
    print(f"测试集 MAE: {test_mae:.6f}")

    return model, history, X_test, y_test, scaler


# 4. 预测函数
def make_predictions(model, last_sequence, scaler, future_steps=1):
    """
    使用训练好的模型进行预测

    参数:
    model: 训练好的模型
    last_sequence: 最后的时间窗口数据（shape: [1, time_steps, 4]）
    scaler: 用于反标准化的scaler
    future_steps: 预测步数
    """
    # 预测
    predictions_scaled = model.predict(last_sequence)

    # 重塑预测结果
    if future_steps == 1:
        predictions_scaled = predictions_scaled.reshape(1, 1, 4)
    else:
        predictions_scaled = predictions_scaled.reshape(1, future_steps, 4)

    # 反标准化
    predictions = scaler.inverse_transform(
        predictions_scaled.reshape(-1, 4)
    ).reshape(future_steps, 4)

    return predictions


# 5. 可视化结果
def plot_results(history, y_true, y_pred, features_names=None):
    """
    可视化训练过程和预测结果

    参数:
    history: 训练历史
    y_true: 真实值
    y_pred: 预测值
    features_names: 特征名称列表
    """
    if features_names is None:
        features_names = [f'Feature {i + 1}' for i in range(4)]

    # 绘制训练历史
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 训练损失
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 绘制每个特征的预测对比
    for i, feature_name in enumerate(features_names):
        row = (i + 1) // 3
        col = (i + 1) % 3

        axes[row, col].plot(y_true[:, i], label='True', alpha=0.7)
        axes[row, col].plot(y_pred[:, i], label='Predicted', alpha=0.7)
        axes[row, col].set_title(f'{feature_name} - True vs Predicted')
        axes[row, col].set_ylabel('Value')
        axes[row, col].set_xlabel('Time Step')
        axes[row, col].legend()
        axes[row, col].grid(True)

    plt.tight_layout()
    plt.show()


# 6. 主函数 - 使用示例
def main():
    """
    主函数示例
    """
    # 示例：创建模拟数据（如果你已经有df，可以跳过这一步）
    # 假设df已经存在，格式为：时间列 + 4个特征列
    np.random.seed(42)
    n_samples = 1000
    time = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')

    # 创建四个相关的时间序列
    t = np.linspace(0, 20, n_samples)
    feature1 = np.sin(t) + np.random.normal(0, 0.1, n_samples)
    feature2 = np.cos(t) + np.random.normal(0, 0.1, n_samples)
    feature3 = t * 0.1 + np.random.normal(0, 0.1, n_samples)
    feature4 = np.sin(t * 2) * np.cos(t) + np.random.normal(0, 0.1, n_samples)

    # 创建DataFrame
    df = pd.DataFrame({
        'time': time,
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4
    })

    print("数据形状:", df.shape)
    print("\n数据预览:")
    print(df.head())

    # 训练模型（单步预测）
    print("\n开始训练单步预测模型...")
    model, history, X_test, y_test, scaler = train_and_predict(
        df,
        time_steps=20,
        future_steps=1,
        test_size=0.2
    )

    # 在测试集上进行预测
    y_pred_scaled = model.predict(X_test)

    # 反标准化
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)

    # 可视化结果
    features_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
    plot_results(history, y_true, y_pred, features_names)

    # 多步预测示例
    print("\n开始训练多步预测模型（预测未来3步）...")
    model_multi, _, _, _, scaler_multi = train_and_predict(
        df,
        time_steps=20,
        future_steps=3,
        test_size=0.2
    )

    # 使用最后的时间窗口进行未来预测
    last_sequence = X_test[-1:].reshape(1, 20, 4)  # 最后一个时间窗口
    future_predictions = make_predictions(
        model_multi,
        last_sequence,
        scaler_multi,
        future_steps=3
    )

    print("\n未来3步预测结果:")
    for i in range(3):
        print(f"第{i + 1}步预测: {future_predictions[i]}")

    return model, model_multi, scaler, scaler_multi


# 7. 实用函数：保存和加载模型
def save_model(model, scaler, model_path='lstm_model.h5', scaler_path='scaler.pkl'):
    """保存模型和scaler"""
    model.save(model_path)
    import joblib
    joblib.dump(scaler, scaler_path)
    print(f"模型已保存到 {model_path}")
    print(f"Scaler已保存到 {scaler_path}")


def load_model(model_path='lstm_model.h5', scaler_path='scaler.pkl'):
    """加载模型和scaler"""
    from tensorflow.keras.models import load_model
    import joblib

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    print(f"模型已从 {model_path} 加载")
    print(f"Scaler已从 {scaler_path} 加载")

    return model, scaler


# 运行主函数
    # 训练模型
model_single, model_multi, scaler_single, scaler_multi = main()

    # 保存模型的示例
    # save_model(model_single, scaler_single, 'lstm_single_step.h5', 'scaler_single.pkl')
    # save_model(model_multi, scaler_multi, 'lstm_multi_step.h5', 'scaler_multi.pkl')


