"""实验 1：GridWorld 环境建模与固定策略的贝尔曼迭代评估。

环境：3×3 网格，障碍 (1, 1)，终点 (2, 2)。
策略：在每个普通状态等概率选择上、下、左、右四个动作。
目标：计算固定策略的状态价值 V^pi，而不是寻找最优策略。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 网格坐标统一使用 (row, col)：左上角为 (0, 0)，向下行号增加，向右列号增加。
ROWS, COLS = 3, 3
WALL = (1, 1)          # 障碍格：不可进入，也不参与价值更新
GOAL = (2, 2)          # 终点格：到达后回合结束，后续价值为 0
STEP_REWARD = -1.0     # 普通移动及撞墙、越界（原地不动）的单步奖励
GOAL_REWARD = 10.0     # 从其他位置进入终点时获得的奖励
GAMMA = 0.9            # 折扣因子
THETA = 1e-10          # 最大价值变化量小于该阈值时停止迭代
MAX_ITERATIONS = 10000 # 防止意外情况下无限循环

# 动作编号：0=上、1=下、2=左、3=右；每项表示 (行变化, 列变化)。
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def step(state, action):
    """执行一个动作，返回 (下一状态, 即时奖励, 是否终止)。

    当前环境为确定性 MDP，所以给定 state 和 action 后结果唯一。
    即使动作导致撞墙或越界，该动作也消耗一个时间步并得到 STEP_REWARD。
    """
    if state == GOAL:
        # 终止状态不再产生新的奖励。
        return GOAL, 0.0, True

    dr, dc = ACTIONS[action]
    nr, nc = state[0] + dr, state[1] + dc

    # 越界或撞上障碍时，智能体保持在原来的状态。
    if not (0 <= nr < ROWS and 0 <= nc < COLS) or (nr, nc) == WALL:
        next_state = state
    else:
        next_state = (nr, nc)

    if next_state == GOAL:
        return next_state, GOAL_REWARD, True
    return next_state, STEP_REWARD, False


def evaluate_uniform_random_policy():
    """使用同步贝尔曼更新，评估四个动作等概率的固定策略。

    数学形式：V_{k+1}(s) = 1/4 * sum_a [r(s,a) + gamma * V_k(s'_a)]。
    因为本环境的转移确定，内层对下一状态的概率求和退化成唯一结果。
    返回：状态价值矩阵、每轮最大变化量、实际迭代轮数。
    """
    values = np.zeros((ROWS, COLS), dtype=float)  # V_0(s)=0
    deltas = []                                    # 记录收敛过程

    for iteration in range(1, MAX_ITERATIONS + 1):
        # 同步更新：本轮所有状态都只能读取旧数组 values。
        new_values = values.copy()

        for row in range(ROWS):
            for col in range(COLS):
                state = (row, col)
                if state in (WALL, GOAL):
                    continue  # 障碍不是可访问状态；终点后续价值固定为 0

                action_values = []
                for action in range(len(ACTIONS)):
                    next_state, reward, _ = step(state, action)

                    # 一个动作对应的贝尔曼备份：r + gamma * V_k(s')。
                    # values[next_state] 使用 NumPy 二维索引 (row, col)。
                    q = reward + GAMMA * values[next_state]
                    action_values.append(q)

                # pi(a|s)=1/4，故四个动作的加权期望就是算术平均。
                # 这是固定策略的期望更新，不是对动作取最大值。
                new_values[state] = np.mean(action_values)

        # 无穷范数误差指标：本轮全部状态估计值的最大绝对变化。
        delta = np.max(np.abs(new_values - values))
        deltas.append(delta)
        values = new_values

        if delta < THETA:
            break
    else:
        print("警告：达到最大迭代次数，尚未达到设定的收敛阈值。")

    return values, np.asarray(deltas), iteration


def plot_result(values, deltas):
    """绘制状态价值热力图与贝尔曼更新变化曲线，并保存到 figures/。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 左图：障碍格没有状态价值，设成 NaN 以区别于数值 0。
    view = values.copy()
    view[WALL] = np.nan
    image = axes[0].imshow(view, cmap="viridis")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    for row in range(ROWS):
        for col in range(COLS):
            state = (row, col)
            if state == WALL:
                label = "WALL"
            elif state == GOAL:
                label = "GOAL"
            else:
                label = f"{values[state]:.2f}"
            axes[0].text(col, row, label, ha="center", va="center",
                         fontsize=10, color="white")

    axes[0].set_title("State values under uniform random policy")
    axes[0].set_xticks(range(COLS))
    axes[0].set_yticks(range(ROWS))
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    # 右图：每轮 max_s |V_{k+1}(s)-V_k(s)|，使用对数纵轴便于观察收敛。
    axes[1].semilogy(np.arange(1, len(deltas) + 1), deltas)
    axes[1].set_title("Bellman iteration residual")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Maximum value change")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "01_gridworld_policy_evaluation.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    values, deltas, iterations = evaluate_uniform_random_policy()
    print(f"收敛迭代轮数：{iterations}；最终最大变化量：{deltas[-1]:.3e}")

    # 障碍格不是有效状态，打印时用 nan 标识。
    display_values = values.copy()
    display_values[WALL] = np.nan
    print("固定随机策略下的状态价值矩阵：")
    print(np.round(display_values, 4))

    print(f"实验图片保存位置：{plot_result(values, deltas)}")
