import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """
    K 臂老虎机（K-Armed Bandit）环境。

    每个动作 a 对应一个 Agent 不可见的真实动作价值：

        q*(a) = E[R | A = a]

    环境初始化时：

        q*(a) ~ N(0, 1)

    当 Agent 执行动作 a 时，实际奖励从下面的分布中采样：

        R ~ N(q*(a), 1)

    Parameters
    ----------
    k : int
        动作数量，即老虎机数量。
    """

    def __init__(self, k: int = 10):
        self.k = k

        # q*(a)：每个动作的真实期望奖励。
        # 这些值属于环境内部信息，Agent 不能直接用于决策。
        self.true_values = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=k,
        )

        # 真实最优动作：
        #
        #     a* = argmax_a q*(a)
        #
        # 该值只用于实验评估，不参与 Agent 的动作选择。
        self.best_action = int(np.argmax(self.true_values))

    def step(self, action: int) -> float:
        """
        执行动作并返回随机奖励。

        Parameters
        ----------
        action : int
            Agent 选择的动作编号。

        Returns
        -------
        float
            本次动作获得的奖励。

        Notes
        -----
        奖励满足：

            R ~ N(q*(action), 1)

        因此，同一个动作每次获得的奖励并不相同，
        只有长期平均奖励会逐渐接近 q*(action)。
        """
        reward = np.random.normal(
            loc=self.true_values[action],
            scale=1.0,
        )

        return float(reward)


class EpsilonGreedyAgent:
    """
    使用 ε-greedy 策略的 Bandit Agent。

    动作选择策略：

        A_t =
            随机动作，          概率 ε
            argmax_a Q(a)，     概率 1 - ε

    动作价值采用样本平均法更新：

        Q(a)
        <- Q(a)
        + 1 / N(a) * [R - Q(a)]

    Parameters
    ----------
    k : int
        动作数量。
    epsilon : float
        ε-greedy 中的探索概率。
    """

    def __init__(self, k: int = 10, epsilon: float = 0.1):
        self.k = k
        self.epsilon = epsilon

        # Q(a)：Agent 对每个动作价值的当前估计。
        self.q_values = np.zeros(k, dtype=float)

        # N(a)：每个动作截至当前被选择的次数。
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
            # Exploration：随机探索一个动作。
            return int(np.random.randint(self.k))

        # Exploitation：选择当前估计价值最高的动作。
        return int(np.argmax(self.q_values))

    def update(self, action: int, reward: float) -> None:
        """
        使用样本平均法增量更新动作价值 Q(a)。

        Parameters
        ----------
        action : int
            本次执行的动作。
        reward : float
            执行动作后获得的奖励。

        Notes
        -----
        数学公式：

            Q(a)
            <- Q(a)
            + 1 / N(a) * [R - Q(a)]

        可以写成统一的增量更新形式：

            New Estimate
            =
            Old Estimate
            +
            Step Size
            *
            Error
        """
        self.action_counts[action] += 1

        count = self.action_counts[action]

        self.q_values[action] += (
            reward - self.q_values[action]
        ) / count


def run_single_bandit(
    k: int,
    epsilon: float,
    steps: int,
):
    """
    运行一次完整的 Bandit 实验。

    一次独立实验包括：

    1. 创建一个新的 Bandit 环境；
    2. 创建一个新的 Agent；
    3. Agent 与环境交互 steps 次；
    4. 记录每一步 reward；
    5. 记录每一步是否选择真实最优动作。

    Parameters
    ----------
    k : int
        动作数量。
    epsilon : float
        ε-greedy 的探索概率。
    steps : int
        单次实验的交互步数。

    Returns
    -------
    rewards : np.ndarray
        shape 为 (steps,)。

        rewards[t] 表示第 t 个时间步获得的奖励。

    optimal_action_flags : np.ndarray
        shape 为 (steps,)。

        optimal_action_flags[t] = 1：
            第 t 步选择了真实最优动作。

        optimal_action_flags[t] = 0：
            第 t 步没有选择真实最优动作。
    """
    bandit = Bandit(k=k)

    agent = EpsilonGreedyAgent(
        k=k,
        epsilon=epsilon,
    )

    rewards = np.zeros(steps, dtype=float)

    optimal_action_flags = np.zeros(
        steps,
        dtype=int,
    )

    for step in range(steps):
        # 1. Agent 根据 ε-greedy 选择动作。
        action = agent.select_action()

        # 2. 环境执行动作并返回随机奖励。
        reward = bandit.step(action)

        # 3. Agent 根据奖励更新 Q(a)。
        agent.update(
            action=action,
            reward=reward,
        )

        # 4. 保存当前时间步的奖励。
        rewards[step] = reward

        # 5. 记录本次是否选择了真实最优动作。
        optimal_action_flags[step] = int(
            action == bandit.best_action
        )

    return rewards, optimal_action_flags


def run_multiple_bandits(
    runs: int = 2000,
    k: int = 10,
    epsilon: float = 0.1,
    steps: int = 1000,
):
    """
    运行多次相互独立的 Bandit 实验。

    每一次 run 都会重新创建：

    - 一个新的 Bandit 环境；
    - 一个新的 Agent。

    因此不同 run 之间：

    - q*(a) 不同；
    - reward 随机序列不同；
    - exploration 时机不同；
    - Agent 的学习轨迹不同。

    Parameters
    ----------
    runs : int
        独立实验次数。
    k : int
        每个 Bandit 的动作数量。
    epsilon : float
        ε-greedy 的探索概率。
    steps : int
        每次独立实验的交互步数。

    Returns
    -------
    average_rewards : np.ndarray
        shape 为 (steps,)。

        average_rewards[t] 表示：
        在第 t 个时间步上，对所有 runs 的 reward 求平均。

    optimal_action_rates : np.ndarray
        shape 为 (steps,)。

        optimal_action_rates[t] 表示：
        在第 t 个时间步上，所有独立实验中选择真实最优动作的比例。
    """

    # rewards_all_runs[i, t]
    # 表示第 i 次独立实验在第 t 步获得的 reward。
    rewards_all_runs = np.zeros(
        (runs, steps),
        dtype=float,
    )

    # optimal_flags_all_runs[i, t]
    # 表示第 i 次独立实验在第 t 步是否选择最优动作。
    #
    # 1：选择最优动作
    # 0：没有选择最优动作
    optimal_flags_all_runs = np.zeros(
        (runs, steps),
        dtype=int,
    )

    for run in range(runs):
        rewards, optimal_flags = run_single_bandit(
            k=k,
            epsilon=epsilon,
            steps=steps,
        )

        rewards_all_runs[run] = rewards
        optimal_flags_all_runs[run] = optimal_flags

    # --------------------------------------------------------
    # axis=0 表示沿着“run 维度”求平均。
    #
    # 原数组 shape：
    #
    #     (runs, steps)
    #
    # 例如：
    #
    #     (2000, 1000)
    #
    # 对 axis=0 求平均后：
    #
    #     (1000,)
    #
    # 即得到每个时间步上的平均结果。
    # --------------------------------------------------------

    average_rewards = np.mean(
        rewards_all_runs,
        axis=0,
    )

    optimal_action_rates = np.mean(
        optimal_flags_all_runs,
        axis=0,
    )

    return average_rewards, optimal_action_rates


def plot_average_reward(
    average_rewards: np.ndarray,
    epsilon: float,
) -> None:
    """
    绘制多次独立实验后的平均奖励曲线。

    Parameters
    ----------
    average_rewards : np.ndarray
        每个时间步上的平均奖励。
    epsilon : float
        当前实验使用的探索概率。
    """
    plt.figure(figsize=(10, 5))

    plt.plot(average_rewards)

    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title(
        f"Average Reward (epsilon={epsilon})"
    )

    plt.tight_layout()
    plt.show()


def plot_optimal_action_rate(
    optimal_action_rates: np.ndarray,
    epsilon: float,
) -> None:
    """
    绘制最优动作选择比例曲线。

    Parameters
    ----------
    optimal_action_rates : np.ndarray
        每个时间步上，选择真实最优动作的比例。
        数值范围为 [0, 1]。
    epsilon : float
        当前实验使用的探索概率。
    """
    plt.figure(figsize=(10, 5))

    # 乘以 100，将比例转换为百分比。
    plt.plot(
        optimal_action_rates * 100
    )

    plt.xlabel("Step")
    plt.ylabel("Optimal Action (%)")
    plt.title(
        f"Optimal Action Rate (epsilon={epsilon})"
    )

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    程序入口。
    """

    # --------------------------------------------------------
    # 实验配置
    # --------------------------------------------------------
    seed = 42

    runs = 2000
    steps = 1000
    k = 10
    epsilon = 0.1

    # 固定随机种子，使整组多次独立实验可复现。
    np.random.seed(seed)

    average_rewards, optimal_action_rates = (
        run_multiple_bandits(
            runs=runs,
            k=k,
            epsilon=epsilon,
            steps=steps,
        )
    )

    print("=" * 60)
    print("Multiple Bandit Experiment")
    print("=" * 60)

    print(f"独立实验次数 runs：{runs}")
    print(f"每次实验步数 steps：{steps}")
    print(f"动作数量 k：{k}")
    print(f"epsilon：{epsilon}")

    print()

    print(
        "最后一个时间步的平均奖励："
        f"{average_rewards[-1]:.4f}"
    )

    print(
        "最后一个时间步的最优动作选择比例："
        f"{optimal_action_rates[-1] * 100:.2f}%"
    )

    print("=" * 60)

    plot_average_reward(
        average_rewards=average_rewards,
        epsilon=epsilon,
    )

    plot_optimal_action_rate(
        optimal_action_rates=optimal_action_rates,
        epsilon=epsilon,
    )


if __name__ == "__main__":
    main()