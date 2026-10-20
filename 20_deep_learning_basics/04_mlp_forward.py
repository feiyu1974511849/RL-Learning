"""
第8讲：MLP前向传播与非线性拟合
运行：python 04_mlp_forward.py
依赖：torch、numpy、matplotlib
输出：figures/04_mlp_fit.png、figures/04_mlp_loss.png

实验分成两部分：
1. 使用手动给定的权重，验证2-3-2网络的前向计算。
2. 使用相同的非线性数据对比线性模型与两隐藏层MLP的拟合效果。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 在没有图形桌面的环境中也能保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def experiment_1_manual_forward():
    """实验一：手动数值推导与nn.Linear前向结果进行对照。"""
    print("\n=== 实验一：2-3-2网络前向传播 ===")
    # 一条样本两个特征。PyTorch习惯使用(batch_size, feature_count)。
    # 因此这里是(1,2)，而不是数学列向量的(2,1)。
    x = torch.tensor([[1.0, 2.0]])

    # 输入维度为2、输出维度为3，因此W1的形状为(3,2)。
    w1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    b1 = torch.tensor([0.0, 0.0, 0.0])
    # 输入维度为3、输出维度为2，因此W2的形状为(2,3)。
    w2 = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    b2 = torch.tensor([0.0, 0.0])

    # 对一批样本，线性层公式是 X @ W.T + b。
    # (1,2)@(2,3)+(3,) -> (1,3)。
    z1 = x @ w1.T + b1
    h = torch.relu(z1)  # ReLU逐元素取max(0,z)，形状不变
    # (1,3)@(3,2)+(2,) -> (1,2)。
    prediction = h @ w2.T + b2

    print("输入shape:", tuple(x.shape), "数值:", x.tolist())
    print("隐藏层线性输出shape:", tuple(z1.shape), "数值:", z1.tolist())
    print("ReLU输出shape:", tuple(h.shape), "数值:", h.tolist())
    print("最终输出shape:", tuple(prediction.shape), "数值:", prediction.tolist())

    # 验证与上节手动推导完全一致：z1=[1,2,-1]，h=[1,2,0]，y=[3,2]。
    assert torch.allclose(z1, torch.tensor([[1., 2., -1.]]))
    assert torch.allclose(h, torch.tensor([[1., 2., 0.]]))
    assert torch.allclose(prediction, torch.tensor([[3., 2.]]))

    # 用真正的nn.Linear重复相同计算；手动设置权重，排除随机初始化影响。
    network = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2))
    with torch.no_grad():
        network[0].weight.copy_(w1)
        network[0].bias.copy_(b1)
        network[2].weight.copy_(w2)
        network[2].bias.copy_(b2)
    automatic_prediction = network(x)
    assert torch.allclose(prediction, automatic_prediction)
    print("nn.Linear计算与手动前向传播一致！")


def make_data():
    """产生非线性回归任务的训练集、验证集和绘图曲线点。"""
    # 通过固定种子，确保同一环境下重复运行具有可比性。
    generator = torch.Generator().manual_seed(SEED)
    # 在[-1,1]上均匀取240个点，使整个区间（包括边界）都有样本。
    # 再随机打乱，避免训练集与验证集按x区间切割。
    x_all = torch.linspace(-1, 1, 240).unsqueeze(1)
    x_all = x_all[torch.randperm(240, generator=generator)]
    # 目标函数：抛物线 + 正弦波。增加一点噪声使任务更贴近真实回归。
    clean_y = x_all.square() + 0.3 * torch.sin(3 * torch.pi * x_all)
    y_all = clean_y + 0.035 * torch.randn((240, 1), generator=generator)

    # 固定随机数据顺序：前200条训练，后40条验证。
    train_x, val_x = x_all[:200], x_all[200:]
    train_y, val_y = y_all[:200], y_all[200:]

    # 绘图时用均匀间隔的点，观察模型拟合出的连续曲线。
    x_plot = torch.linspace(-1, 1, 401).unsqueeze(1)
    y_true = x_plot.square() + 0.3 * torch.sin(3 * torch.pi * x_plot)
    return train_x, train_y, val_x, val_y, x_plot, y_true


class LinearModel(nn.Module):
    """对照组：没有隐藏层的线性模型。"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)


class MLPModel(nn.Module):
    """实验组：1->32->32->1，两层隐藏层均使用ReLU。"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),   # (B,1)->(B,32)
            nn.ReLU(),          # 引入非线性，形状不变
            nn.Linear(32, 32),  # (B,32)->(B,32)
            nn.ReLU(),
            nn.Linear(32, 1),   # (B,32)->(B,1)
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model, x_train, y_train, x_val, y_val, epochs=700):
    """使用相同的训练流程比较模型，返回训练/验证MSE历史。"""
    criterion = nn.MSELoss()  # 计算一批样本的均方误差
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history_train, history_val = [], []

    for epoch in range(epochs):
        # 训练模式：对于含Dropout/BatchNorm的网络尤其重要。
        model.train()
        pred = model(x_train)  # 全批量训练，单个Epoch更新一次
        loss = criterion(pred, y_train)

        optimizer.zero_grad()  # 不让梯度叠加上一个Epoch的结果
        loss.backward()       # Autograd计算每个参数的梯度
        optimizer.step()      # Adam根据梯度更新参数

        # 验证不参与参数更新，也不需要存储梯度计算图。
        model.eval()
        with torch.no_grad():
            train_mse = criterion(model(x_train), y_train).item()
            val_mse = criterion(model(x_val), y_val).item()
        history_train.append(train_mse)
        history_val.append(val_mse)

        if (epoch + 1) % 175 == 0:
            print(f"{model.__class__.__name__} | Epoch {epoch + 1:>3} | "
                  f"训练MSE={train_mse:.5f} | 验证MSE={val_mse:.5f}")
    return np.array(history_train), np.array(history_val)


def experiment_2_nonlinear_fit():
    """实验二：比较线性模型与MLP拟合非线性函数的能力，并保存图表。"""
    print("\n=== 实验二：线性模型与MLP非线性拟合对比 ===")
    torch.manual_seed(SEED)
    x_train, y_train, x_val, y_val, x_plot, y_true = make_data()
    linear = LinearModel()
    mlp = MLPModel()

    print("训练集输入：", tuple(x_train.shape), "验证集输入：", tuple(x_val.shape))
    print("线性模型参数量：", sum(p.numel() for p in linear.parameters()))
    print("MLP参数量：", sum(p.numel() for p in mlp.parameters()))
    print("说明：本实验使用全批量梯度下降，每个Epoch只有一次参数更新。")

    linear_train, linear_val = train_model(linear, x_train, y_train, x_val, y_val)
    mlp_train, mlp_val = train_model(mlp, x_train, y_train, x_val, y_val)

    # 绘制预测曲线：真实关系、线性模型、MLP、带噪声训练样本。
    with torch.no_grad():
        linear_pred = linear(x_plot).squeeze(1).numpy()
        mlp_pred = mlp(x_plot).squeeze(1).numpy()
    x_np = x_plot.squeeze(1).numpy()
    true_np = y_true.squeeze(1).numpy()

    plt.figure(figsize=(9, 5.5))
    plt.scatter(x_train.squeeze(1).numpy(), y_train.squeeze(1).numpy(),
                s=15, alpha=0.4, label="Training samples")
    plt.plot(x_np, true_np, linewidth=2.5, label="True nonlinear function")
    plt.plot(x_np, linear_pred, linewidth=2, label="Linear model")
    plt.plot(x_np, mlp_pred, linewidth=2, label="MLP (ReLU)")
    plt.xlabel("Input x")
    plt.ylabel("Target / prediction")
    plt.title("Nonlinear regression: Linear vs MLP")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    fit_path = FIGURE_DIR / "04_mlp_fit.png"
    plt.savefig(fit_path, dpi=180)
    plt.close()

    # 训练/验证损失：对数坐标便于观察数量级差异。
    plt.figure(figsize=(9, 5.5))
    plt.plot(linear_train, label="Linear - train")
    plt.plot(linear_val, linestyle="--", label="Linear - validation")
    plt.plot(mlp_train, label="MLP - train")
    plt.plot(mlp_val, linestyle="--", label="MLP - validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error (log scale)")
    plt.title("Training and validation losses")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    loss_path = FIGURE_DIR / "04_mlp_loss.png"
    plt.savefig(loss_path, dpi=180)
    plt.close()

    print(f"\n最终验证 MSE：线性模型 {linear_val[-1]:.6f}；MLP {mlp_val[-1]:.6f}")
    print("已保存拟合对比图：", fit_path)
    print("已保存损失对比图：", loss_path)


def main():
    experiment_1_manual_forward()
    experiment_2_nonlinear_fit()


if __name__ == "__main__":
    main()
