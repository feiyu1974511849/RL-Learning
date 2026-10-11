import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class NonStationaryBandit:
    """
    非平稳 K 臂老虎机（Non-stationary K-Armed Bandit）。

    每个动作的真实动作价值会随时间发生随机游走：

        q*_{t+1}(a) = q*_t(a) + noise

    其中：

        noise ~ N(0, random_walk_std^2)

    Parameters
    ----------
    k : int
        动作数量。
    random_walk_std : float
        每一步真实动作价值随机变化的标准差。
    """

    def __init__(self, k: int = 10, random_walk_std: float = 0.01):
        self.k = k
        self.random_walk_std = random_walk_std

        # 初始真实动作价值 q*_0(a)。
        self.true_values = np.zeros(k, dtype=float)

    def step(self, action: int) -> float:
        """
        执行动作并返回奖励，同时让环境发生一次随机游走。

        Parameters
        ----------
        action : int
            Agent 选择的动作编号。

        Returns
        -------
        float
            当前动作获得的随机奖励。
        """

        # 当前奖励：R_t ~ N(q*_t(action), 1)。
        reward = np.random.normal(
            loc=self.true_values[action],
            scale=1.0,
        )

        # 每个动作的真实价值都发生一个小的随机变化。
        self.true_values += np.random.normal(
            loc=0.0,
            scale=self.random_walk_std,
            size=self.k,
        )

        return float(reward)

    def get_best_action(self) -> int:
        """
        返回当前时刻的真实最优动作。

        Returns
        -------
        int
            当前 q*(a) 最大的动作编号。
        """
        return int(np.argmax(self.true_values))


class SampleAverageAgent:
    """
    使用样本平均法（Sample-Average Method）的 ε-greedy Agent。

    动作价值更新：

        Q(a) <- Q(a) + 1 / N(a) * [R - Q(a)]

    Parameters
    ----------
    k : int
        动作数量。
    epsilon : float
        ε-greedy 的探索概率。
    """

    def __init__(self, k: int = 10, epsilon: float = 0.1):
        self.k = k
        self.epsilon = epsilon

        self.q_values = np.zeros(k, dtype=float)
        self.action_counts = np.zeros(k, dtype=int)

    def select_action(self) -> int:
        """根据 ε-greedy 策略选择动作。"""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.k))

        return int(np.argmax(self.q_values))

    def update(self, action: int, reward: float) -> None:
        """
        使用样本平均法更新 Q(a)。

        Q(a) <- Q(a) + 1 / N(a) * [R - Q(a)]
        """
        self.action_counts[action] += 1

        self.q_values[action] += (
            reward - self.q_values[action]
        ) / self.action_counts[action]


class ConstantStepAgent:
    """
    使用固定步长（Constant Step Size）的 ε-greedy Agent。

    动作价值更新：

        Q(a) <- Q(a) + alpha * [R - Q(a)]

    Parameters
    ----------
    k : int
        动作数量。
    epsilon : float
        ε-greedy 的探索概率。
    alpha : float
        固定步长。
    """

    def __init__(
        self,
        k: int = 10,
        epsilon: float = 0.1,
        alpha: float = 0.1,
    ):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha

        self.q_values = np.zeros(k, dtype=float)

    def select_action(self) -> int:
        """根据 ε-greedy 策略选择动作。"""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.k))

        return int(np.argmax(self.q_values))

    def update(self, action: int, reward: float) -> None:
        """
        使用固定步长更新 Q(a)。

        Q(a) <- Q(a) + alpha * [R - Q(a)]
        """
        self.q_values[action] += (
            self.alpha
            * (reward - self.q_values[action])
        )


def run_single_experiment(
    agent_type: str,
    k: int,
    epsilon: float,
    alpha: float,
    steps: int,
    random_walk_std: float,
):
    """
    运行一次非平稳 Bandit 实验。

    Parameters
    ----------
    agent_type : str
        "sample_average" 或 "constant_step"。
    k : int
        动作数量。
    epsilon : float
        探索概率。
    alpha : float
        固定步长，仅 ConstantStepAgent 使用。
    steps : int
        交互步数。
    random_walk_std : float
        环境随机游走的标准差。

    Returns
    -------
    rewards : np.ndarray
        每一步获得的奖励。
    optimal_flags : np.ndarray
        每一步是否选择当前真实最优动作。
    """
    bandit = NonStationaryBandit(
        k=k,
        random_walk_std=random_walk_std,
    )

    if agent_type == "sample_average":
        agent = SampleAverageAgent(
            k=k,
            epsilon=epsilon,
        )
    elif agent_type == "constant_step":
        agent = ConstantStepAgent(
            k=k,
            epsilon=epsilon,
            alpha=alpha,
        )
    else:
        raise ValueError(
            "agent_type 必须是 'sample_average' 或 'constant_step'"
        )

    rewards = np.zeros(steps, dtype=float)
    optimal_flags = np.zeros(steps, dtype=int)

    for step in range(steps):
        # 当前最优动作必须在环境随机游走前记录。
        best_action = bandit.get_best_action()

        action = agent.select_action()
        reward = bandit.step(action)

        agent.update(
            action=action,
            reward=reward,
        )

        rewards[step] = reward
        optimal_flags[step] = int(
            action == best_action
        )

    return rewards, optimal_flags


def run_multiple_experiments(
    agent_type: str,
    runs: int,
    k: int,
    epsilon: float,
    alpha: float,
    steps: int,
    random_walk_std: float,
):
    """
    对指定 Agent 运行多次独立的非平稳 Bandit 实验。

    Returns
    -------
    average_rewards : np.ndarray
        每个时间步上的平均奖励。
    optimal_action_rates : np.ndarray
        每个时间步选择当前真实最优动作的比例。
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
        rewards, optimal_flags = run_single_experiment(
            agent_type=agent_type,
            k=k,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_walk_std=random_walk_std,
        )

        rewards_all_runs[run] = rewards
        optimal_flags_all_runs[run] = optimal_flags

    average_rewards = np.mean(
        rewards_all_runs,
        axis=0,
    )

    optimal_action_rates = np.mean(
        optimal_flags_all_runs,
        axis=0,
    )

    return average_rewards, optimal_action_rates


def run_comparison(
    runs: int,
    k: int,
    epsilon: float,
    alpha: float,
    steps: int,
    random_walk_std: float,
):
    """
    比较样本平均法与固定步长法。
    """
    results = {}

    print("Running Sample Average ...")

    average_rewards, optimal_action_rates = (
        run_multiple_experiments(
            agent_type="sample_average",
            runs=runs,
            k=k,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_walk_std=random_walk_std,
        )
    )

    results["Sample Average"] = {
        "average_rewards": average_rewards,
        "optimal_action_rates": optimal_action_rates,
    }

    print("Running Constant Step Size ...")

    average_rewards, optimal_action_rates = (
        run_multiple_experiments(
            agent_type="constant_step",
            runs=runs,
            k=k,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_walk_std=random_walk_std,
        )
    )

    results["Constant Step Size"] = {
        "average_rewards": average_rewards,
        "optimal_action_rates": optimal_action_rates,
    }

    return results


def plot_results(results: dict) -> None:
    """在一张图中并排绘制两项指标，并保存到脚本所在目录的 figures 文件夹。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, result in results.items():
        axes[0].plot(result["average_rewards"], label=name)
        axes[1].plot(result["optimal_action_rates"] * 100, label=name)

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
    fig.suptitle("Non-stationary Bandit")
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "figures" / "05_bandit_nonstationary.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Figure saved to: {output_path}")
    plt.show()


def print_final_results(results: dict) -> None:
    """输出最后一个时间步的实验结果。"""
    print()
    print("=" * 72)
    print("Final Results")
    print("=" * 72)

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

    print("=" * 72)


def main() -> None:
    """程序入口。"""

    seed = 42

    k = 10
    runs = 2000
    steps = 10000

    epsilon = 0.1
    alpha = 0.1

    random_walk_std = 0.01

    np.random.seed(seed)

    results = run_comparison(
        runs=runs,
        k=k,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        random_walk_std=random_walk_std,
    )

    print_final_results(results)

    plot_results(results)


if __name__ == "__main__":
    main()