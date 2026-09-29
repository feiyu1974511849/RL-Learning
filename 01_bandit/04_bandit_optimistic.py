import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """
    K 臂老虎机（K-Armed Bandit）环境。

    每个动作 a 对应一个 Agent 不可见的真实动作价值：

        q*(a) = E[R | A = a]

    环境初始化时：

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

        # q*(a)：每个动作的真实期望奖励。
        self.true_values = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=k,
        )

        # a* = argmax_a q*(a)，仅用于实验评价。
        self.best_action = int(
            np.argmax(self.true_values)
        )

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
            从 N(q*(action), 1) 中采样得到的奖励。
        """
        reward = np.random.normal(
            loc=self.true_values[action],
            scale=1.0,
        )

        return float(reward)


class EpsilonGreedyAgent:
    """
    使用固定步长更新的 ε-greedy Agent。

    动作选择：

        以 ε 的概率随机探索；
        以 1-ε 的概率选择当前 Q(a) 最大的动作。

    动作价值更新：

        Q(a)
        <- Q(a)
        + alpha * [R - Q(a)]

    Parameters
    ----------
    k : int
        动作数量。
    epsilon : float
        探索概率 ε。
    alpha : float
        固定步长。
    initial_value : float
        Q(a) 的初始值。
    """

    def __init__(
        self,
        k: int = 10,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        initial_value: float = 0.0,
    ):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha

        # 所有动作的 Q(a) 使用相同初始值。
        self.q_values = np.full(
            k,
            initial_value,
            dtype=float,
        )

    def select_action(self) -> int:
        """
        根据 ε-greedy 策略选择动作。

        Returns
        -------
        int
            选择的动作编号。
        """
        if np.random.random() < self.epsilon:
            return int(
                np.random.randint(self.k)
            )

        return int(
            np.argmax(self.q_values)
        )

    def update(
        self,
        action: int,
        reward: float,
    ) -> None:
        """
        使用固定步长更新 Q(a)。

        Parameters
        ----------
        action : int
            本次执行的动作。
        reward : float
            执行动作后获得的奖励。

        Notes
        -----
        更新公式：

            Q(a)
            <- Q(a)
            + alpha * [R - Q(a)]

        其中：

            alpha

        不会随着动作被访问次数增加而减小。
        """
        self.q_values[action] += (
            self.alpha
            * (
                reward
                - self.q_values[action]
            )
        )


def run_single_bandit(
    k: int,
    epsilon: float,
    alpha: float,
    initial_value: float,
    steps: int,
):
    """
    运行一次独立 Bandit 实验。

    Parameters
    ----------
    k : int
        动作数量。
    epsilon : float
        ε-greedy 探索概率。
    alpha : float
        固定步长。
    initial_value : float
        Q(a) 初始值。
    steps : int
        单次实验交互步数。

    Returns
    -------
    rewards : np.ndarray
        每个时间步获得的奖励。
    optimal_flags : np.ndarray
        每个时间步是否选择真实最优动作。
    """
    bandit = Bandit(k=k)

    agent = EpsilonGreedyAgent(
        k=k,
        epsilon=epsilon,
        alpha=alpha,
        initial_value=initial_value,
    )

    rewards = np.zeros(
        steps,
        dtype=float,
    )

    optimal_flags = np.zeros(
        steps,
        dtype=int,
    )

    for step in range(steps):
        action = agent.select_action()

        reward = bandit.step(action)

        agent.update(
            action=action,
            reward=reward,
        )

        rewards[step] = reward

        optimal_flags[step] = int(
            action == bandit.best_action
        )

    return rewards, optimal_flags


def run_multiple_bandits(
    runs: int,
    k: int,
    epsilon: float,
    alpha: float,
    initial_value: float,
    steps: int,
):
    """
    运行多次独立 Bandit 实验。

    Parameters
    ----------
    runs : int
        独立实验次数。
    k : int
        动作数量。
    epsilon : float
        ε-greedy 探索概率。
    alpha : float
        固定步长。
    initial_value : float
        Q(a) 初始值。
    steps : int
        每次实验的交互步数。

    Returns
    -------
    average_rewards : np.ndarray
        每个时间步上的平均奖励。
    optimal_action_rates : np.ndarray
        每个时间步选择真实最优动作的比例。
    """
    rewards_all_runs = np.zeros(
        (runs, steps),
        dtype=float,
    )

    optimal_flags_all_runs = np.zeros(
        (runs, steps),
        dtype=int,
    )

    for run in range(runs):
        rewards, optimal_flags = run_single_bandit(
            k=k,
            epsilon=epsilon,
            alpha=alpha,
            initial_value=initial_value,
            steps=steps,
        )

        rewards_all_runs[run] = rewards

        optimal_flags_all_runs[run] = (
            optimal_flags
        )

    average_rewards = np.mean(
        rewards_all_runs,
        axis=0,
    )

    optimal_action_rates = np.mean(
        optimal_flags_all_runs,
        axis=0,
    )

    return (
        average_rewards,
        optimal_action_rates,
    )


def run_comparison(
    runs: int,
    k: int,
    steps: int,
    alpha: float,
):
    """
    比较普通 ε-greedy 与 Optimistic Greedy。

    Returns
    -------
    results : dict
        保存两种方法的实验结果。
    """

    results = {}

    # --------------------------------------------------------
    # 方法 A：
    # 普通 ε-greedy
    #
    # Q0(a) = 0
    # epsilon = 0.1
    # --------------------------------------------------------
    average_rewards, optimal_action_rates = (
        run_multiple_bandits(
            runs=runs,
            k=k,
            epsilon=0.1,
            alpha=alpha,
            initial_value=0.0,
            steps=steps,
        )
    )

    results["epsilon-greedy"] = {
        "average_rewards": average_rewards,
        "optimal_action_rates": optimal_action_rates,
    }

    # --------------------------------------------------------
    # 方法 B：
    # Optimistic Greedy
    #
    # Q0(a) = 5
    # epsilon = 0
    # --------------------------------------------------------
    average_rewards, optimal_action_rates = (
        run_multiple_bandits(
            runs=runs,
            k=k,
            epsilon=0.0,
            alpha=alpha,
            initial_value=5.0,
            steps=steps,
        )
    )

    results["optimistic-greedy"] = {
        "average_rewards": average_rewards,
        "optimal_action_rates": optimal_action_rates,
    }

    return results


def plot_average_reward(
    results: dict,
) -> None:
    """
    绘制两种方法的 Average Reward 曲线。
    """
    plt.figure(figsize=(10, 5))

    for name, result in results.items():
        plt.plot(
            result["average_rewards"],
            label=name,
        )

    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title(
        "Average Reward: "
        "Epsilon-Greedy vs Optimistic Greedy"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_optimal_action_rate(
    results: dict,
) -> None:
    """
    绘制两种方法的 Optimal Action Rate 曲线。
    """
    plt.figure(figsize=(10, 5))

    for name, result in results.items():
        plt.plot(
            result["optimal_action_rates"] * 100,
            label=name,
        )

    plt.xlabel("Step")
    plt.ylabel("Optimal Action (%)")
    plt.title(
        "Optimal Action Rate: "
        "Epsilon-Greedy vs Optimistic Greedy"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


def print_final_results(
    results: dict,
) -> None:
    """
    输出两种方法在最后一个时间步的实验结果。
    """
    print()
    print("=" * 70)
    print("Final Results")
    print("=" * 70)

    for name, result in results.items():
        final_reward = result[
            "average_rewards"
        ][-1]

        final_optimal_rate = result[
            "optimal_action_rates"
        ][-1]

        print(
            f"{name:<20} | "
            f"Average Reward={final_reward:.4f} | "
            f"Optimal Action={final_optimal_rate * 100:.2f}%"
        )

    print("=" * 70)


def main() -> None:
    """
    程序入口。
    """

    seed = 42

    k = 10
    runs = 2000
    steps = 1000

    alpha = 0.1

    np.random.seed(seed)

    results = run_comparison(
        runs=runs,
        k=k,
        steps=steps,
        alpha=alpha,
    )

    print_final_results(results)

    plot_average_reward(results)

    plot_optimal_action_rate(results)


if __name__ == "__main__":
    main()