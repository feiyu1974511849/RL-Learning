"""
实验 03：使用 PyTorch 训练线性回归模型

学习目标：
1. 使用模拟数据构造回归任务，理解 X/y 的张量形状。
2. 使用 TensorDataset 和 DataLoader 进行小批量训练。
3. 掌握 nn.Linear、MSELoss、SGD、反向传播与优化器的配合。
4. 学习训练集/验证集划分、model.train()/eval()、no_grad()。
5. 绘制并保存损失曲线与预测对比图；保存模型参数。

运行：python 03_deep_learning_basics/03_linear_regression.py
依赖：torch、numpy、matplotlib
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 在无图形界面的服务器上也能正常保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 输出文件的位置以当前脚本为准，避免因终端工作目录不同而保存到错误位置。
BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / 'figures'
MODEL_DIR = BASE_DIR / 'models'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42  # 固定随机种子，使每次运行的模拟数据和初始化尽量一致
NUM_SAMPLES = 1000
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.05


def prepare_data():
    """创建模拟数据，将其分成互不重叠的训练集和验证集。

    x1、x2 是模拟的、已经按便于训练的尺度表示的特征。
    真实关系为 y = 3*x1 - 2*x2 + 1.5 + 随机噪声。
    虽然我们知道真实系数，但训练时不把它们直接交给模型。
    """
    torch.manual_seed(SEED)

    # 特征矩阵：1000 行（样本数）× 2 列（特征数）。
    X = torch.randn(NUM_SAMPLES, 2, dtype=torch.float32)

    # 切片 X[:, 0:1] 保留二维形状 (1000, 1)，避免标签维度不匹配。
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    noise = 0.2 * torch.randn(NUM_SAMPLES, 1)
    y = 3.0 * x1 - 2.0 * x2 + 1.5 + noise

    # 随机划分数据；使用同一组索引，保证输入和标签始终一一对应。
    indices = torch.randperm(NUM_SAMPLES)
    train_size = int(NUM_SAMPLES * TRAIN_RATIO)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    # TensorDataset：让第 i 个特征样本始终对应第 i 个标签。
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # 训练集打乱顺序，验证集保持固定顺序，便于绘图比较。
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('数据总量：', NUM_SAMPLES)
    print('训练集形状：', tuple(X_train.shape), tuple(y_train.shape))
    print('验证集形状：', tuple(X_val.shape), tuple(y_val.shape))
    print('训练批次数：', len(train_loader))
    print('首个训练批次的形状：', tuple(next(iter(train_loader))[0].shape))
    return train_loader, val_loader, X_val, y_val


class LinearRegression(nn.Module):
    """只有一个线性层的回归模型：输入2个特征，输出1个数值。"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        # x: (batch_size, 2)
        # 内部计算：x @ weight.T + bias
        # weight: (1, 2)，bias: (1,)
        # 输出: (batch_size, 1)
        return self.linear(x)


def train_model(model, train_loader, val_loader):
    """对模型训练若干轮，并记录训练和验证集的平均 MSE。"""
    criterion = nn.MSELoss()  # 均方误差：对当前批次所有元素求平均
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    train_history, val_history = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()  # 训练模式；对本实验线性层数值无影响，但应养成习惯
        train_loss_sum = 0.0
        train_count = 0

        for X_batch, y_batch in train_loader:
            # 前向传播：(batch, 2) -> (batch, 1)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # 先清零旧梯度，然后反向传播，再更新参数。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 乘以实际批次大小，避免最后不足32条的批次被等权处理。
            batch_count = X_batch.shape[0]
            train_loss_sum += loss.item() * batch_count
            train_count += batch_count

        train_mse = train_loss_sum / train_count
        train_history.append(train_mse)

        # 验证集只用来衡量泛化表现，不参与参数更新。
        model.eval()  # 评估模式
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():  # 不需要建立反向求导图，减少额外开销
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                batch_count = X_batch.shape[0]
                val_loss_sum += loss.item() * batch_count
                val_count += batch_count

        val_mse = val_loss_sum / val_count
        val_history.append(val_mse)

        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch:3d}/{EPOCHS} | 训练MSE={train_mse:.5f} | 验证MSE={val_mse:.5f}')

    return train_history, val_history


def save_figures(train_history, val_history, model, X_val, y_val):
    """绘制两个有明确坐标轴和标题的实验图，并保存到 figures 目录。"""
    plt.rcParams['axes.unicode_minus'] = False

    # 图一：训练集与验证集损失随 Epoch 的变化。
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(train_history) + 1)
    ax.plot(epochs, train_history, label='Train MSE', linewidth=1.8)
    ax.plot(epochs, val_history, label='Validation MSE', linewidth=1.8)
    ax.set(xlabel='Epoch', ylabel='Mean Squared Error', title='Linear Regression: Training and Validation Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    loss_file = FIGURE_DIR / '03_linear_regression_loss.png'
    fig.savefig(loss_file, dpi=160)
    plt.close(fig)

    # 图二：验证集真实值与预测值，理想情况落在 y=x 对角线附近。
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val).flatten().cpu().numpy()
    actual = y_val.flatten().cpu().numpy()

    lower = float(min(actual.min(), y_pred.min()))
    upper = float(max(actual.max(), y_pred.max()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, y_pred, s=18, alpha=0.65, label='Validation samples')
    ax.plot([lower, upper], [lower, upper], linestyle='--', linewidth=1.5, label='Ideal: prediction = target')
    ax.set(xlabel='True target', ylabel='Model prediction', title='Linear Regression: Predictions vs Targets')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    pred_file = FIGURE_DIR / '03_linear_regression_predictions.png'
    fig.savefig(pred_file, dpi=160)
    plt.close(fig)

    print('损失曲线已保存：', loss_file)
    print('预测对比图已保存：', pred_file)


def main():
    train_loader, val_loader, X_val, y_val = prepare_data()
    model = LinearRegression()
    print('\n模型结构：\n', model)
    print('权重形状：', tuple(model.linear.weight.shape))
    print('偏置形状：', tuple(model.linear.bias.shape))
    print('训练前的权重：', model.linear.weight.detach().flatten().tolist())
    print('训练前的偏置：', model.linear.bias.detach().item())

    train_history, val_history = train_model(model, train_loader, val_loader)

    learned_w = model.linear.weight.detach().flatten().tolist()
    learned_b = model.linear.bias.detach().item()
    print('\n真实参数：w1=3.0000, w2=-2.0000, b=1.5000')
    print(f'训练参数：w1={learned_w[0]:.4f}, w2={learned_w[1]:.4f}, b={learned_b:.4f}')
    print(f'最后一轮训练MSE={train_history[-1]:.5f}, 验证MSE={val_history[-1]:.5f}')

    # 仅保存模型参数，供后续 load_state_dict() 使用。
    model_file = MODEL_DIR / '03_linear_regression.pth'
    torch.save(model.state_dict(), model_file)
    print('模型参数已保存：', model_file)
    save_figures(train_history, val_history, model, X_val, y_val)


if __name__ == '__main__':
    main()
