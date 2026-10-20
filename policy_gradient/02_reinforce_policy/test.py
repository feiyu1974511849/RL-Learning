import gymnasium as gym
import torch
from model import Agent


def test_render(agent, env, episodes=5):
    """使用训练好的策略网络测试智能体，并渲染游戏画面。"""

    for episode in range(episodes):
        # 重置环境，获取初始状态
        state, info = env.reset()
        done = False
        total_reward = 0

        # 执行动作，直到当前回合结束
        while not done:
            env.render()

            # 根据当前状态选择动作
            action, _ = agent.get_action(state)

            # 执行动作，获取下一状态、奖励和终止信息
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 更新环境状态并累计奖励
            state = next_state
            total_reward += reward

        print(f"回合 {episode + 1}: 总奖励 = {total_reward}")

    # 测试结束后关闭环境
    env.close()


# 创建测试环境，启用可视化渲染
env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=200)

# 创建智能体并加载训练好的策略网络参数
agent = Agent()
agent.pi.load_state_dict(torch.load("policy_model.pth", weights_only=True))

# 切换到评估模式
agent.pi.eval()

# 测试5个回合
test_render(agent, env, episodes=5)