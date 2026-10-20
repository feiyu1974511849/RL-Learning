"""

实验目标：
1. 对比低容量线性模型、小型 MLP、大型 MLP 的拟合能力。
2. 通过训练/验证损失、预测曲线观察欠拟合和过拟合。
3. 对比 L2 权重衰减（weight_decay）对同一大型网络的影响。
4. 了解为什么需要训练集、验证集和独立测试集。

运行：python 05_overfitting_regularization.py
依赖：torch、numpy、matplotlib。
输出：在脚本所在目录 figures/ 下创建两张实验图片。
"""
from pathlib import Path
import copy
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

SEED = 42
EPOCHS = 1000
LR = 0.01
OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int):
    """固定 Python、NumPy、PyTorch 随机种子，减少运行结果波动。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def true_function(x: torch.Tensor) -> torch.Tensor:
    """模拟真实的数据规律：平滑的非线性曲线。"""
    return torch.sin(2 * torch.pi * x) + 0.25 * x


def make_data():
    """构建小训练集与独立验证、测试集：训练样本稀疏且包含较强噪声。"""
    # 注意：训练集只有 18 条样本，容易出现高容量模型记住噪声的现象。
    x_train = torch.linspace(-1, 1, 18).unsqueeze(1)
    y_train = true_function(x_train) + 0.35 * torch.randn_like(x_train)

    # 验证集、测试集用于观察未见数据上的误差；不参与梯度更新。
    x_val = torch.linspace(-0.98, 0.98, 100).unsqueeze(1)
    y_val = true_function(x_val) + 0.10 * torch.randn_like(x_val)
    x_test = torch.linspace(-0.99, 0.99, 120).unsqueeze(1)
    y_test = true_function(x_test) + 0.10 * torch.randn_like(x_test)
    return x_train, y_train, x_val, y_val, x_test, y_test


def build_model(kind: str) -> nn.Module:
    """构建不同容量网络。线性模型 2 个参数，小 MLP 较少，大 MLP 较多。"""
    if kind == 'linear':
        return nn.Linear(1, 1)
    if kind == 'small':
        return nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
    if kind == 'large':
        return nn.Sequential(
            nn.Linear(1, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )
    raise ValueError(f'不支持的网络类型：{kind}')


def train_model(model, x_train, y_train, x_val, y_val, weight_decay=0.0):
    """执行一套完整训练。注意：验证集绝不能调用 backward/step。"""
    # MSE 是预测与标签之差的平方均值。
    criterion = nn.MSELoss()
    # Adam 根据梯度更新参数；weight_decay>0 时引入 L2 权重衰减。
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    history = {'train': [], 'val': []}
    for epoch in range(EPOCHS):
        model.train()  # 设置训练模式（如存在 Dropout、BatchNorm 会影响行为）。
        predictions = model(x_train)  # 前向传播，(18,1)->(18,1)。
        loss = criterion(predictions, y_train)  # 当前训练集损失。
        optimizer.zero_grad()  # 清空上一步的旧梯度，防止自动累加。
        loss.backward()  # 反向传播：计算所有可学习参数的梯度。
        optimizer.step()  # 优化器实际更新参数。

        model.eval()  # 切换到评估模式。
        with torch.no_grad():  # 验证时无需记录自动求导图。
            # 同一个 epoch 的训练损失，用更新后的参数重新计算，保证两条曲线时点一致。
            train_loss = criterion(model(x_train), y_train).item()
            val_loss = criterion(model(x_val), y_val).item()
        history['train'].append(train_loss)
        history['val'].append(val_loss)
    return history


def main():
    seed_everything(SEED)
    x_train, y_train, x_val, y_val, x_test, y_test = make_data()
    models = {}
    histories = {}
    configs = [
        ('Linear (low capacity)', 'linear', 0.0),
        ('Small MLP', 'small', 0.0),
        ('Large MLP', 'large', 0.0),
        ('Large MLP + L2', 'large', 0.0005),
    ]
    for i, (name, kind, decay) in enumerate(configs):
        # 固定网络初始化；两个大型 MLP 初始参数相同，有助于公平比较正则化。
        seed_everything(SEED + (2 if kind == 'large' else i))
        model = build_model(kind)
        histories[name] = train_model(model, x_train, y_train, x_val, y_val, decay)
        models[name] = model
        n_params = sum(p.numel() for p in model.parameters())
        print(f'{name}: 参数数={n_params}, '
              f'训练MSE={histories[name]["train"][-1]:.5f}, '
              f'验证MSE={histories[name]["val"][-1]:.5f}')

    # 图1：训练点、真实函数和各种模型的预测曲线。
    x_plot = torch.linspace(-1.0, 1.0, 400).unsqueeze(1)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.scatter(x_train.numpy(), y_train.numpy(), s=35, alpha=0.8,
               label='Training samples (noisy)')
    ax.plot(x_plot.numpy(), true_function(x_plot).numpy(), linewidth=2.5,
            label='True function')
    with torch.no_grad():
        for name, model in models.items():
            ax.plot(x_plot.numpy(), model(x_plot).numpy(), linewidth=1.8, label=name)
    ax.set(xlabel='Input x', ylabel='Target / prediction',
           title='Model capacity and regularization: function fitting')
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / '05_capacity_fit.png', dpi=180)
    plt.close(fig)

    # 图2：四个模型各自训练曲线、验证曲线。
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    for ax, (name, _, _) in zip(axes.ravel(), configs):
        hist = histories[name]
        ax.plot(hist['train'], label='Train MSE', linewidth=1.7)
        ax.plot(hist['val'], label='Validation MSE', linewidth=1.7)
        ax.set_yscale('log')  # 纵轴对数刻度，便于比较不同数量级的误差。
        ax.set_title(name)
        ax.set_ylabel('MSE (log scale)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('Epoch')
    fig.suptitle('Train vs validation loss: underfitting / overfitting', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / '05_capacity_losses.png', dpi=180)
    plt.close(fig)

    print('\n独立测试集（仅训练结束后查看）：')
    with torch.no_grad():
        for name, model in models.items():
            mse = nn.functional.mse_loss(model(x_test), y_test).item()
            print(f'{name}: Test MSE={mse:.5f}')
    print(f'\n图像保存至：{OUT}')

if __name__ == '__main__':
    main()
