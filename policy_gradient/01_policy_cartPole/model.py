import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    """策略神经网络，根据环境状态输出动作概率。"""

    def __init__(self, action_size=2):
        super().__init__()

        # 状态维度为4，隐藏层维度为128，输出维度为动作数量
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        """前向传播，计算各个动作的概率。"""
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class Agent:
    """智能体类，负责选择动作、保存经验和更新策略网络。"""

    def __init__(self):
        self.gamma = 0.98       # 折扣因子
        self.lr = 0.0002        # 学习率
        self.action_size = 2    # 动作空间大小

        self.memory = []        # 存储每一步的奖励和动作概率
        self.pi = Policy(self.action_size)  # 初始化策略神经网络
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state):
        """根据当前状态，按照策略网络输出的概率选择动作。"""

        # 将状态转换为张量，形状从(4,)变为(1, 4)
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)

        # 计算动作概率分布
        probs = self.pi(state)
        probs = probs[0]

        # 根据概率分布随机采样动作
        m = Categorical(probs)
        action = m.sample().item()

        # 返回动作及其对应的概率
        return action, probs[action]

    def add(self, reward, prob):
        """保存当前时间步的奖励和动作概率。"""
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        """根据整条轨迹的折扣回报更新策略网络。"""
        G, loss = 0, 0

        # 逆序遍历奖励，计算整条轨迹的折扣回报
        for reward, _ in reversed(self.memory):
            G = reward + self.gamma * G

        # 使用相同的折扣回报计算每一步的策略损失
        for _, prob in self.memory:
            loss += -prob.log() * G

        # 清空梯度，反向传播并更新网络参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空当前回合保存的经验
        self.memory = []