
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class BanditAgent:
    """实现 ε-Greedy 和 UCB 两种动作选择策略。"""

    def __init__(self, k=10, method="epsilon_greedy", epsilon=0.1, c=2.0):
        self.k = k
        self.method = method
        self.epsilon = epsilon
        self.c = c

        self.q_values = np.zeros(k)
        self.action_counts = np.zeros(k, dtype=int)

    def select_action(self, t, rng):
        """根据指定策略选择动作。"""
        if self.method == "epsilon_greedy":
            if rng.random() < self.epsilon:
                return int(rng.integers(self.k))

            # 随机处理多个动作估计值相同的情况
            best_actions = np.flatnonzero(
                self.q_values == np.max(self.q_values)
            )
            return int(rng.choice(best_actions))

        if self.method == "ucb":
            # 对尚未尝试的动作优先探索，避免 N(a)=0
            untried_actions = np.flatnonzero(self.action_counts == 0)
            if len(untried_actions) > 0:
                return int(rng.choice(untried_actions))

            # UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))
            exploration_bonus = self.c * np.sqrt(
                np.log(t) / self.action_counts
            )
            ucb_values = self.q_values + exploration_bonus

            best_actions = np.flatnonzero(
                ucb_values == np.max(ucb_values)
            )
            return int(rng.choice(best_actions))

        raise ValueError(f"Unknown method: {self.method}")

    def update(self, action, reward):
        """使用样本平均增量公式更新动作价值估计。"""
        self.action_counts[action] += 1
        self.q_values[action] += (
            reward - self.q_values[action]
        ) / self.action_counts[action]


def run_experiment(
    method,
    true_values_all,
    steps=1000,
    epsilon=0.1,
    c=2.0,
    seed=42,
):
    """对指定算法进行多次独立实验，返回两个评价指标。"""
    rng = np.random.default_rng(seed)

    runs, k = true_values_all.shape
    rewards = np.zeros((runs, steps))
    optimal_actions = np.zeros((runs, steps))

    for run in range(runs):
        true_values = true_values_all[run]
        best_action = int(np.argmax(true_values))

        agent = BanditAgent(
            k=k,
            method=method,
            epsilon=epsilon,
            c=c,
        )

        for step in range(steps):
            # t 从 1 开始，访问次数统计当前动作执行前的数据
            t = step + 1
            action = agent.select_action(t, rng)

            # 平稳环境：真实动作价值固定，奖励围绕真实价值随机波动
            reward = rng.normal(loc=true_values[action], scale=1.0)

            agent.update(action, reward)

            rewards[run, step] = reward
            optimal_actions[run, step] = int(action == best_action)

    average_rewards = np.mean(rewards, axis=0)
    optimal_action_rates = np.mean(optimal_actions, axis=0)

    return average_rewards, optimal_action_rates


def main():
    k = 10
    runs = 2000
    steps = 1000
    seed = 42

    # 所有算法使用同一批 Bandit 环境，保证环境难度一致
    environment_rng = np.random.default_rng(seed)
    true_values_all = environment_rng.normal(
        loc=0.0,
        scale=1.0,
        size=(runs, k),
    )

    configs = {
        "Epsilon-Greedy (epsilon=0.1)": {
            "method": "epsilon_greedy",
            "epsilon": 0.1,
        },
        "UCB (c=1)": {
            "method": "ucb",
            "c": 1.0,
        },
        "UCB (c=2)": {
            "method": "ucb",
            "c": 2.0,
        },
        "UCB (c=4)": {
            "method": "ucb",
            "c": 4.0,
        },
    }

    results = {}

    for index, (name, config) in enumerate(configs.items()):
        print(f"Running: {name}")

        avg_rewards, optimal_rates = run_experiment(
            method=config["method"],
            true_values_all=true_values_all,
            steps=steps,
            epsilon=config.get("epsilon", 0.1),
            c=config.get("c", 2.0),
            seed=seed + index + 1,
        )

        results[name] = (avg_rewards, optimal_rates)

        # 汇总最后 100 步，减少单个时间步随机波动的影响
        final_reward = np.mean(avg_rewards[-100:])
        final_optimal_rate = np.mean(optimal_rates[-100:])

        print(f"  Average Reward: {final_reward:.4f}")
        print(f"  Optimal Action Rate: {final_optimal_rate:.2%}")

    # 绘制平均奖励与最优动作选择率
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, (avg_rewards, optimal_rates) in results.items():
        axes[0].plot(avg_rewards, label=name)
        axes[1].plot(optimal_rates * 100, label=name)

    axes[0].set_title("Average Reward")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Average Reward")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Optimal Action Rate")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Optimal Action (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    output_path = (
    Path(__file__).resolve().parent
    / "figures"
    / "06_bandit_ucb.png"
)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
