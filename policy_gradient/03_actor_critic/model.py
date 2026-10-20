import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)  # 输出的是一个评分（标量值）

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98  # 折扣因子
        self.lr_pi = 0.0002  # 策略网络的学习率
        self.lr_v = 0.0005  # 价值网络的学习率
        self.action_size = 2

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :])  # 增加小批量的轴
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        # 增加小批量的轴
        # S_t
        state = torch.tensor(state[np.newaxis, :])
        # S_{t+1}
        next_state = torch.tensor(next_state[np.newaxis, :])

        # ①self.v 的损失：均方差
        # target = R_t + γ*V_ω(S_{t+1})
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.detach()
        # V_ω(S_t)
        v = self.v(state)
        loss_fn = nn.MSELoss()
        loss_v = loss_fn(v, target)

        # ②self.pi 的损失
        # δ = R_t + γ*V_ω(S_{t+1}) - V_ω(S_t)
        delta = target - v
        loss_pi = -torch.log(action_prob) * delta.item()

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()
