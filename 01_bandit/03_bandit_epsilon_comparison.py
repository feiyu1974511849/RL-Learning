import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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

        # q*(a)：环境内部的真实动作价值。
        # Agent 不能直接使用这些值进行动作选择。
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
    ε-greedy Bandit Agent。

    动作选择：

        以 ε 的概率随机探索；
        以 1-ε 的概率选择当前 Q(a) 最大的动作。

    动作价值使用样本平均法更新：

        Q(a) <- Q(a) + 1/N(a) * [R - Q(a)]

    Parameters
    ----------
    k : int
        动作数量。
    epsilon : float
        探索概率。
    """

    def __init__(
        self,
        k: int = 10,
        epsilon: float = 0.1,
    ):
        self.k = k
        self.epsilon = epsilon

        # Q(a)：Agent 对各动作价值的估计。
        self.q_values = np.zeros(
            k,
            dtype=float,
        )

        # N(a)：各动作累计被选择的次数。
        self.action_counts = np.zeros(
            k,
            dtype=int,
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
            # Exploration：随机选择一个动作。
            return int(
                np.random.randint(self.k)
            )

        # Exploitation：选择当前估计价值最高的动作。
        return int(
            np.argmax(self.q_values)
        )

    def update(
        self,
        action: int,
        reward: float,
    ) -> None:
        """
        使用样本平均法增量更新 Q(a)。

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
            + 1/N(a) * [R - Q(a)]

        其中：

            1/N(a)

        是当前样本平均估计对应的步长。
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
    运行一次独立 Bandit 实验。

    Parameters
    ----------
    k : int
        动作数量。
    epsilon : float
        ε-greedy 探索概率。
    steps : int
        交互步数。

    Returns
    -------
    rewards : np.ndarray
        每个时间步获得的奖励，shape=(steps,)。
    optimal_flags : np.ndarray
        每个时间步是否选择真实最优动作，shape=(steps,)。
    """
    bandit = Bandit(k=k)

    agent = EpsilonGreedyAgent(
        k=k,
        epsilon=epsilon,
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
    steps: int,
):
    """
    对指定 ε 运行多次独立 Bandit 实验。

    Parameters
    ----------
    runs : int
        独立实验次数。
    k : int
        动作数量。
    epsilon : float
        当前测试的探索概率。
    steps : int
        每次独立实验的交互步数。

    Returns
    -------
    average_rewards : np.ndarray
        每个时间步在所有独立实验上的平均奖励。
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
            steps=steps,
        )

        rewards_all_runs[run] = rewards
        optimal_flags_all_runs[run] = optimal_flags

    # axis=0：对 run 这一维求平均，保留 step 这一维。
    average_rewards = np.mean(
        rewards_all_runs,
        axis=0,
    )

    optimal_action_rates = np.mean(
        optimal_flags_all_runs,
        axis=0,
    )

    return average_rewards, optimal_action_rates


def run_epsilon_comparison(
    epsilons: list[float],
    runs: int,
    k: int,
    steps: int,
):
    """
    对多个 ε 分别执行 Bandit 实验。

    Parameters
    ----------
    epsilons : list[float]
        需要比较的 ε 值。
    runs : int
        每个 ε 对应的独立实验次数。
    k : int
        动作数量。
    steps : int
        每次实验的交互步数。

    Returns
    -------
    results : dict
        保存每个 ε 对应的实验结果。

        数据结构：

            results[epsilon]["average_rewards"]
            results[epsilon]["optimal_action_rates"]
    """
    results = {}

    for epsilon in epsilons:
        print(
            f"Running epsilon={epsilon} ..."
        )

        average_rewards, optimal_action_rates = (
            run_multiple_bandits(
                runs=runs,
                k=k,
                epsilon=epsilon,
                steps=steps,
            )
        )

        results[epsilon] = {
            "average_rewards": average_rewards,
            "optimal_action_rates": optimal_action_rates,
        }

    return results


def plot_results(results: dict) -> None:
    """在一张图中并排绘制两项指标，并保存到脚本所在目录的 figures 文件夹。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for epsilon, result in results.items():
        label = f"epsilon={epsilon}"
        axes[0].plot(result["average_rewards"], label=label)
        axes[1].plot(result["optimal_action_rates"] * 100, label=label)

    axes[0].set_title("Average Reward")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Average Reward")
    axes[1].set_title("Optimal Action Rate")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Optimal Action (%)")
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Epsilon Comparison")
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "figures" / "03_bandit_epsilon_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Figure saved to: {output_path}")
    plt.show()


def print_final_results(
    results: dict,
) -> None:
    """
    输出不同 ε 在最后一个时间步的实验结果。
    """
    print()
    print("=" * 65)
    print("Final Results")
    print("=" * 65)

    for epsilon, result in results.items():
        final_reward = result[
            "average_rewards"
        ][-1]

        final_optimal_rate = result[
            "optimal_action_rates"
        ][-1]

        print(
            f"epsilon={epsilon:<4} | "
            f"Average Reward={final_reward:.4f} | "
            f"Optimal Action={final_optimal_rate * 100:.2f}%"
        )

    print("=" * 65)


def main() -> None:
    """
    程序入口。
    """

    # 实验配置
    seed = 42

    k = 10
    runs = 2000
    steps = 1000

    epsilons = [
        0.0,
        0.01,
        0.1,
    ]

    # 固定整组实验的随机序列，使实验结果可以复现。
    np.random.seed(seed)

    results = run_epsilon_comparison(
        epsilons=epsilons,
        runs=runs,
        k=k,
        steps=steps,
    )

    print_final_results(results)

    plot_results(results)


if __name__ == "__main__":
    main()