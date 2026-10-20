import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from model import Agent


# 创建倒立摆环境和智能体，保持原始环境的200步限制
env = gym.make("CartPole-v1", max_episode_steps=200)
agent = Agent()
reward_history = []

for episode in range(3000):
    # 重置环境，获取初始状态
    state, info = env.reset()
    done = False
    total_reward = 0

    # 采样一条完整轨迹
    while not done:
        # 根据当前状态选择动作
        action, prob = agent.get_action(state)

        # 执行动作，获取下一状态、奖励及终止信息
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 保存当前时间步的奖励和动作概率
        agent.add(reward, prob)

        # 更新环境状态并累积奖励
        state = next_state
        total_reward += reward

    # 当前回合结束后，使用反向传播更新策略神经网络
    agent.update()
    reward_history.append(total_reward)

    # 每100个回合打印一次训练结果
    if episode % 100 == 0:
        print("回合:{}, 总奖励:{:.1f}".format(episode, total_reward))

# 保存训练完成的策略神经网络参数
torch.save(agent.pi.state_dict(), "policy_model.pth")
print("模型已保存")

# 关闭环境
env.close()

# 绘制每个回合的奖励变化曲线
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode")
plt.grid(True)
plt.show()