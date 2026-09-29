import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """
    K 臂老虎机（K-Armed Bandit）环境。

    每个动作 a 对应一个 Agent 不可见的真实动作价值：

        q*(a) = E[R | A = a]

    初始化时：
        q*(a) ~ N(0, 1)

    执行动作 a 时：
        R ~ N(q*(a), 1)

    Parameters
    ----------
    k : int
        动作数量。
    """

    def __init__(self, k: int = 10):
        self.k = k

        # q*(a)：环境内部真实动作价值，Agent 不可直接用于决策。
        self.true_values = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=k,
        )

        # a* = argmax_a q*(a)，仅用于实验评估。
        self.best_action = int(np.argmax(self.true_values))

    def step(self, action: int) -> float:
        """
        执行动作并返回随机奖励。

        Parameters
        ----------
        action : int
            动作编号。

        Returns
        -------
        float
            从 N(q*(action), 1) 中采样得到的奖励。
        """
        reward = np.random.normal(
            loc=self.true_values[action],
            scale=1.0,
        )
        return float(reward)


class EpsilonGreedyAgent:
    """
    使用 ε-greedy 策略和样本平均法的 Bandit Agent。

    动作选择：

        A_t =
            随机动作，            概率 ε
            argmax_a Q(a)，       概率 1 - ε

    动作价值更新：

        Q(a) <- Q(a) + 1 / N(a) * [R - Q(a)]

    Parameters
    ----------
    k : int
        动作数量。
    epsilon : float
        探索概率 ε。
    """

    def __init__(self, k: int = 10, epsilon: float = 0.1):
        self.k = k
        self.epsilon = epsilon

        # Q(a)：Agent 对各动作价值的估计。
        self.q_values = np.zeros(k, dtype=float)

        # N(a)：各动作累计被选择的次数。
        self.action_counts = np.zeros(k, dtype=int)

    def select_action(self) -> int:
        """
        根据 ε-greedy 策略选择动作。

        Returns
        -------
        int
            选择的动作编号。
        """
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.k))

        return int(np.argmax(self.q_values))

    def update(self, action: int, reward: float) -> None:
        """
        使用样本平均法增量更新 Q(a)。

        更新公式：

            Q(a) <- Q(a) + 1 / N(a) * [R - Q(a)]

        Parameters
        ----------
        action : int
            本次执行的动作。
        reward : float
            执行动作后获得的奖励。
        """
        self.action_counts[action] += 1

        count = self.action_counts[action]

        # 增量式样本平均：
        # New Estimate = Old Estimate + Step Size * Error
        self.q_values[action] += (
            reward - self.q_values[action]
        ) / count


def run_experiment(
    k: int = 10,
    epsilon: float = 0.1,
    steps: int = 1000,
):
    """
    运行一次 K-Armed Bandit 实验。

    Parameters
    ----------
    k : int
        动作数量。
    epsilon : float
        ε-greedy 的探索概率。
    steps : int
        与环境交互的总步数。

    Returns
    -------
    rewards : np.ndarray
        每一步获得的奖励。
    optimal_action_flags : np.ndarray
        每一步是否选择真实最优动作。
        选择最优动作为 1，否则为 0。
    agent : EpsilonGreedyAgent
        实验结束后的 Agent。
    bandit : Bandit
        本次实验使用的环境。
    """
    bandit = Bandit(k=k)
    agent = EpsilonGreedyAgent(
        k=k,
        epsilon=epsilon,
    )

    # 已知实验长度时，直接预分配数组比不断 append 更清晰。
    rewards = np.zeros(steps, dtype=float)
    optimal_action_flags = np.zeros(steps, dtype=int)

    for step in range(steps):
        action = agent.select_action()
        reward = bandit.step(action)

        agent.update(
            action=action,
            reward=reward,
        )

        rewards[step] = reward
        optimal_action_flags[step] = int(
            action == bandit.best_action
        )

    return (
        rewards,
        optimal_action_flags,
        agent,
        bandit,
    )


def print_experiment_result(
    agent: EpsilonGreedyAgent,
    bandit: Bandit,
    optimal_action_flags: np.ndarray,
) -> None:
    """
    输出单次实验的主要结果。
    """
    optimal_action_rate = np.mean(optimal_action_flags)

    print("=" * 60)
    print("真实动作价值 q*(a)：")
    print(bandit.true_values)

    print("\nAgent 估计动作价值 Q(a)：")
    print(agent.q_values)

    print("\n各动作选择次数 N(a)：")
    print(agent.action_counts)

    print(f"\n真实最优动作：{bandit.best_action}")
    print(
        "Agent 估计的最优动作："
        f"{np.argmax(agent.q_values)}"
    )
    print(
        "最优动作选择比例："
        f"{optimal_action_rate * 100:.2f}%"
    )
    print("=" * 60)


def plot_rewards(
    rewards: np.ndarray,
    epsilon: float,
) -> None:
    """
    绘制单次实验的原始 Reward 曲线。

    注意：
    单步 Reward 含有较强随机噪声，因此曲线通常会明显波动。
    后续会使用 Moving Average 和多次独立实验进一步分析。
    """
    plt.figure(figsize=(10, 5))

    plt.plot(rewards)

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(
        f"Reward over Time (epsilon={epsilon})"
    )

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    程序入口。
    """
    seed = 1
    k = 10
    epsilon = 0.1
    steps = 1000

    # 固定随机种子，保证当前实验可复现。
    np.random.seed(seed)

    rewards, optimal_flags, agent, bandit = run_experiment(
        k=k,
        epsilon=epsilon,
        steps=steps,
    )

    print_experiment_result(
        agent=agent,
        bandit=bandit,
        optimal_action_flags=optimal_flags,
    )

    plot_rewards(
        rewards=rewards,
        epsilon=epsilon,
    )


if __name__ == "__main__":
    main()