import numpy as np

# ============================================================
# 1. 定义 Q-table
# ============================================================
#
# Q-table 可以理解成一张表：
#
#             action
#           0    1    2
# state 0  1.0  4.0  2.0
# state 1  5.0  2.0  3.0
# state 2  2.0  1.0  7.0
# state 3  3.0  6.0  4.0
#
# Q(s, a) 表示：
# 在状态 state=s 时执行 action=a 的价值。
#
# 当前共有：
# 4 个 state
# 3 个 action

Q = np.array([
    [1.0, 4.0, 2.0],
    [5.0, 2.0, 3.0],
    [2.0, 1.0, 7.0],
    [3.0, 6.0, 4.0]
])

# ============================================================
# 2. 任务 A：查看 Q-table 的 shape
# ============================================================

print("Q-table:")
print(Q)
print("\nQ.shape: ", Q.shape)

# ============================================================
# 3. 任务 B：找到某个状态下最大的 Q-value 和对应动作
# ============================================================

# Q[state] 取出 Q-table 的第 state 行。
# state = 2 时：
# Q[2] = [2.0, 1.0, 7.0]
state = 2

q_values = Q[state]
print("\nstate =", state)
print("Q[state] =", q_values)

# np.max() 返回最大的“值”
# 数学上对应：max_a Q(s, a)
best_value = np.max(q_values)

# np.argmax() 返回最大值所在的“下标”，下标正好就是 action。
# 数学上对应 argmax_a Q(s, a)

best_action = np.argmax(q_values)


print("best_value =", best_value)
print("best_action =", best_action)

# ============================================================
# 4. 任务 C：实现 ε-greedy（ε-贪心）策略
# ============================================================

def epsilon_greedy(Q, state, epsilon):
    """
    使用 ε-greedy 策略选择动作。
    参数：
        Q:
            Q-table。
            Q[s, a] 表示状态 s 下执行动作 a 的价值。

        state:
            当前状态。

        epsilon:
            探索概率 ε。

            例如 epsilon = 0.1：
            大约 10% 的时候进行随机探索；
            大约 90% 的时候选择当前认为最好的动作。

    返回：
        action:
            最终选择的动作。
    """

    if np.random.random() < epsilon:
        # -------------------------
        # Exploration：探索
        # -------------------------

        # Q.shape[1] 表示动作数量。
        # 当前：
        # Q.shape = (4, 3)
        # 所以：
        # Q.shape[1] = 3
        # np.random.randint(3)随机产生：
        # 0、1、2
        # 即随机选择一个动作。
        action = np.random.randint(Q.shape[1])
    else:
        # -------------------------
        # Exploitation：利用
        # -------------------------
        # Q[state]：
        # 当前状态下所有动作的 Q-value。
        # np.argmax(Q[state])：
        # 找到 Q-value 最大的动作。
        # 数学上：
        # a* = argmax_a Q(s, a)
        action = np.argmax(Q[state])
    return action

# ============================================================
# 5. 简单测试 ε-greedy
# ============================================================

state = 2
epsilon = 0.1

action = epsilon_greedy(Q, state, epsilon)

print("\n一次 ε-greedy 选择：")
print("action =", action)

# 对于 state = 2：
# Q[2] = [2.0, 1.0, 7.0]
# 所以当前贪心动作是：
# action = 2
# 因此绝大多数时候输出 2。
# 但是因为 epsilon = 0.1，
# 偶尔会进行随机探索，所以也可能输出 0 或 1。


# ============================================================
# 6. 任务 D：重复实验 10,000 次
# ============================================================
# 我们想验证：ε-greedy 的实际动作选择概率是否和理论分析一致。

num_trials = 10000

# counts 用来统计每个动作被选择了多少次。所以建立：
# [0, 0, 0]

counts = np.zeros(Q.shape[1], dtype=int)

for _ in range(num_trials):

    action = epsilon_greedy(Q, state, epsilon)

    counts[action] += 1


# ============================================================
# 7. 输出实验结果
# ============================================================

print("\n===== 10,000 次实验结果 =====")

for action, count in enumerate(counts):

    # 实验概率：
    # 某动作被选择的次数 / 总实验次数

    probability = count / num_trials

    print(
        f"action {action}: "
        f"{count} 次, "
        f"概率约为 {probability:.4f}"
    )


# ============================================================
# 8. 理论概率分析
# ============================================================
# 当前：state = 2
# Q[2] = [2.0, 1.0, 7.0]
# 所以贪心动作：action 2
# epsilon = 0.1有 90% 的概率直接利用：P(利用) = 0.9
# 利用时一定选择 action 2。
# 另外有 10% 的概率探索：P(探索) = 0.1
# 探索时三个动作均匀随机选择。所以探索过程中：
# P(action 0 | 探索) = 1/3
# P(action 1 | 探索) = 1/3
# P(action 2 | 探索) = 1/3
# 因此：P(action 0) = 0.1 × 1/3 ≈ 0.0333


num_actions = Q.shape[1]

exploration_probability = epsilon / num_actions

greedy_action = np.argmax(Q[state])


print("\n===== 理论概率 =====")

for action in range(num_actions):

    if action == greedy_action:

        # 贪心动作：
        #
        # 利用时会选择它
        # +
        # 探索时也可能随机选择到它

        probability = (
            (1 - epsilon)
            + exploration_probability
        )

    else:

        # 非贪心动作只有探索的时候
        # 才可能被随机选到。

        probability = exploration_probability

    print(
        f"action {action}: "
        f"理论概率约为 {probability:.4f}"
    )