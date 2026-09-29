# 第 1 章 强化学习基础与多臂老虎机（Multi-Armed Bandit, MAB）阶段笔记

> 本笔记整理截至“乐观初始值（Optimistic Initial Values）”实验结束时已经学习的知识。  
> 公式统一使用 Markdown + LaTeX 语法，建议使用支持数学公式的 Markdown 编辑器（如 Typora、Obsidian、VS Code + Markdown Preview Enhanced）查看。

---

## 1. 多臂老虎机问题

### 1.1 基本定义

$k$ 臂老虎机（$k$-Armed Bandit）包含 $k$ 个可选动作：

$$
a \in \{0,1,\ldots,k-1\}
$$

智能体（Agent）在每个时间步 $t$ 选择一个动作 $A_t$，环境（Environment）返回一个奖励 $R_t$。

Bandit 与后续完整强化学习问题相比非常简单：

- 没有状态转移；
- 每一步只需要选择动作；
- 核心问题是如何在“探索”和“利用”之间进行权衡。

---

## 2. 真实动作价值与动作价值估计

### 2.1 真实动作价值（True Action Value）

动作 $a$ 的真实动作价值定义为：

$$
q_*(a)=\mathbb{E}[R_t \mid A_t=a]
$$

含义：

> 当选择动作 $a$ 时，能够获得的奖励的期望值。

在当前实验中，每个动作的真实价值初始化为：

$$
q_*(a)\sim\mathcal{N}(0,1)
$$

执行动作 $a$ 后，实际奖励满足：

$$
R_t\sim\mathcal{N}(q_*(a),1)
$$

因此，即使选择同一个动作，每次获得的奖励也可能不同。

---

### 2.2 最优动作（Optimal Action）

真实最优动作：

$$
a_* = \arg\max_a q_*(a)
$$

在代码中：

```python
best_action = np.argmax(true_values)
```

注意：

- $q_*(a)$ 属于环境内部的真实信息；
- Agent 不能直接使用 $q_*(a)$ 选择动作；
- `best_action` 只用于实验评价。

---

### 2.3 动作价值估计（Action-Value Estimate）

Agent 无法直接知道 $q_*(a)$，因此维护估计：

$$
Q_t(a)\approx q_*(a)
$$

代码中通常表示为：

```python
q_values[action]
```

目标是通过不断与环境交互，让：

$$
Q_t(a)\rightarrow q_*(a)
$$

---

## 3. 样本平均法（Sample-Average Method）

假设动作 $a$ 已经被选择 $N(a)$ 次，对应奖励为：

$$
R_1,R_2,\ldots,R_{N(a)}
$$

最直接的估计方法是：

$$
Q(a)=\frac{1}{N(a)}\sum_{i=1}^{N(a)}R_i
$$

即使用历史奖励的样本平均值估计真实动作价值。

---

### 3.1 增量更新形式

不需要每次保存所有历史奖励，可以使用：

$$
Q(a)
\leftarrow
Q(a)
+
\frac{1}{N(a)}
\left[R-Q(a)\right]
$$

其中：

- $Q(a)$：旧估计；
- $R$：最新奖励；
- $R-Q(a)$：估计误差；
- $\frac{1}{N(a)}$：步长（Step Size）。

可以抽象成一个非常重要的形式：

$$
\boxed{
\text{新估计}
=
\text{旧估计}
+
\text{步长}
\times
(\text{目标}-\text{旧估计})
}
$$

这个结构以后会在 TD、Q-learning 等算法中反复出现。

代码对应：

```python
q_values[action] += (
    reward - q_values[action]
) / action_counts[action]
```

---

## 4. 探索与利用（Exploration vs. Exploitation）

### 4.1 利用（Exploitation）

利用当前已有知识，选择估计价值最高的动作：

$$
A_t=\arg\max_a Q_t(a)
$$

优点：

- 当前看来能够获得较高奖励。

问题：

- 当前的 $Q(a)$ 可能并不准确；
- 可能因为早期随机奖励而长期停留在次优动作。

---

### 4.2 探索（Exploration）

尝试当前并不一定最优的动作，以获得更多信息。

探索可能降低当前一步的奖励，但能够帮助 Agent：

- 获取其他动作的信息；
- 修正错误的价值估计；
- 发现之前没有发现的更优动作。

因此强化学习中存在：

$$
\boxed{
\text{探索（Exploration）}
\quad\text{vs.}\quad
\text{利用（Exploitation）}
}
$$

---

## 5. $\epsilon$-贪心策略（$\epsilon$-Greedy Policy）

$\epsilon$-贪心策略：

$$
A_t=
\begin{cases}
\text{随机选择动作}, & \text{概率 }\epsilon\\
\arg\max_a Q_t(a), & \text{概率 }1-\epsilon
\end{cases}
$$

代码：

```python
if np.random.random() < epsilon:
    action = np.random.randint(k)
else:
    action = np.argmax(q_values)
```

注意：

> 探索时必须先执行动作，之后环境才返回奖励。不能先查看奖励，再决定是否真正执行这个动作。

Bandit 的交互顺序是：

$$
A_t \rightarrow R_t
$$

而不是：

$$
\text{先观察所有奖励} \rightarrow A_t
$$

---

### 5.1 为什么探索动作即使不好也必须接受奖励？

因为奖励只有在执行动作以后才能观察。

例如：

```python
reward = bandit.step(action)
```

执行完成后，这个时间步已经发生，不能因为奖励较低而“撤销”动作。

而且单次奖励本身带有随机噪声：

$$
R_t\sim\mathcal{N}(q_*(a),1)
$$

一次较低奖励不能证明该动作的真实价值一定较低。

---

## 6. $\epsilon$-Greedy 选择最优动作的概率

假设：

- $k=10$；
- $\epsilon=0.1$；
- Agent 已经正确识别最优动作，即

$$
\arg\max_a Q(a)=a_*
$$

利用阶段选择最优动作的概率：

$$
1-\epsilon=0.9
$$

探索阶段随机选择 10 个动作之一，仍可能碰巧选中最优动作：

$$
\epsilon\frac{1}{k}
=
0.1\times\frac{1}{10}
=
0.01
$$

因此：

$$
P(A_t=a_*)
=
(1-\epsilon)+\frac{\epsilon}{k}
$$

代入：

$$
P(A_t=a_*)=0.9+0.01=0.91
$$

即：

$$
\boxed{91\%}
$$

注意：

> 91% 的前提是 Agent 已经正确识别真正的最优动作。

如果：

$$
\arg\max_a Q(a)\neq a_*
$$

实际最优动作选择率会更低。

---

## 7. 随机种子（Random Seed）

NumPy 中：

```python
np.random.seed(42)
```

用于固定伪随机数生成器产生的随机序列，使实验具有可复现性（Reproducibility）。

在 Bandit 实验中，随机数会影响：

- $q_*(a)$ 的生成；
- 每次获得的 reward；
- $\epsilon$-greedy 是否触发探索；
- 探索时选择哪个动作；
- 后续 $Q(a)$ 的学习轨迹。

因此不同 seed 对应不同的随机实验轨迹。

不能通过挑选表现最好的 seed 来评价算法。

---

## 8. 单次实验与多次独立实验

### 8.1 单次实验（Single Run）

例如：

$$
1\text{ 个 Bandit}\times1000\text{ steps}
$$

得到一条随机奖励序列：

$$
R_1,R_2,\ldots,R_{1000}
$$

单次实验容易受到随机性的显著影响。

---

### 8.2 多次独立实验（Multiple Independent Runs）

例如：

$$
2000\text{ runs}\times1000\text{ steps}
$$

每个 run 都重新创建：

- 一个新的 Bandit；
- 一组新的 $q_*(a)$；
- 一个新的 Agent；
- 一条新的随机交互轨迹。

实验数组：

```python
rewards_all_runs.shape == (2000, 1000)
```

可以理解为：

```text
                step
           0    1    2    ... 999
run 0      R    R    R    ...  R
run 1      R    R    R    ...  R
run 2      R    R    R    ...  R
...
run 1999   R    R    R    ...  R
```

---

## 9. Average Reward

第 $t$ 个时间步的平均奖励：

$$
\bar{R}_t
=
\frac{1}{M}
\sum_{i=1}^{M}
R_t^{(i)}
$$

其中 $M$ 为独立实验次数。

代码：

```python
average_rewards = np.mean(
    rewards_all_runs,
    axis=0,
)
```

如果原数组：

```text
shape = (2000, 1000)
```

沿 `axis=0` 求平均后：

```text
shape = (1000,)
```

含义：

> 对 run 维度求平均，保留 step 维度。

---

## 10. Optimal Action Rate

定义指示变量：

$$
I_t^{(i)}
=
\begin{cases}
1, & A_t^{(i)}=a_*^{(i)}\\
0, & A_t^{(i)}\neq a_*^{(i)}
\end{cases}
$$

第 $t$ 步的最优动作选择比例：

$$
\text{OptimalActionRate}_t
=
\frac{1}{M}
\sum_{i=1}^{M}
I_t^{(i)}
$$

代码：

```python
optimal_action_rates = np.mean(
    optimal_flags_all_runs,
    axis=0,
)
```

例如第 1000 步：

```text
Optimal Action = 80%
```

表示：

> 在第 1000 个时间步，所有独立实验中约 80% 的 Agent 选择了各自环境里的真实最优动作。

它不是“某一个 Agent 在整个 1000 步中有 80% 的动作是最优动作”。

---

## 11. Moving Average 与跨 Run 平均的区别

移动平均（Moving Average）：

> 在同一个 run 内，对邻近时间步进行平均，用于平滑时间序列。

多次独立实验平均：

> 在同一个时间步，对多个独立 run 的结果求平均。

二者不是同一个统计量。

---

## 12. 不同 $\epsilon$ 的实验

已经比较：

$$
\epsilon\in\{0,\ 0.01,\ 0.1\}
$$

实验设置：

$$
2000\text{ runs}\times1000\text{ steps}
$$

观察到的核心现象如下。

### 12.1 $\epsilon=0$：纯贪心（Greedy）

没有主动随机探索：

$$
A_t=\arg\max_a Q_t(a)
$$

早期随机奖励可能使某个次优动作暂时具有较高 $Q(a)$。

随后：

```text
早期随机结果
    ↓
某个动作暂时看起来最好
    ↓
不断利用该动作
    ↓
其他动作缺乏采样
    ↓
难以发现真正的最优动作
```

这种现象可以理解为过早利用（Premature Exploitation）。

---

### 12.2 $\epsilon=0.01$

具有少量持续探索。

优点：

- 能够纠正部分早期错误；
- 相比纯 greedy 更有机会发现真正最优动作。

缺点：

- 探索速度较慢。

对某个指定动作，通过随机探索选中的概率为：

$$
0.01\times\frac1{10}=0.001
$$

即 $0.1\%$。

---

### 12.3 $\epsilon=0.1$

探索更加充分。

对某个指定动作，通过随机探索选中的概率：

$$
0.1\times\frac1{10}=0.01
$$

即 $1\%$。

在当前实验设置下，它比 $\epsilon=0$ 和 $\epsilon=0.01$ 更快发现较好的动作。

但不能据此推出：

$$
\text{“}\epsilon\text{ 越大越好”}
$$

如果：

$$
\epsilon=1
$$

Agent 将完全随机选择动作，即使已经学到了哪个动作最好，也不会利用这些知识。

---

## 13. 乐观初始值（Optimistic Initial Values）

普通初始化：

$$
Q_0(a)=0
$$

乐观初始化故意设置：

$$
Q_0(a)=5
$$

虽然真实动作价值通常远低于 5，但这种高估是有意的。

核心思想：

> 未充分尝试的动作暂时被认为“可能非常好”，从而吸引 greedy 策略主动尝试。

---

### 13.1 为什么 $\epsilon=0$ 仍然可以产生探索？

假设：

$$
Q=[5,5,5]
$$

并使用纯 greedy：

$$
A_t=\arg\max_a Q(a)
$$

尝试 action 0 后，若其估计下降：

$$
Q=[4.6,5,5]
$$

那么 greedy 下一步会选择仍保持高估值的其他动作。

因此：

$$
\boxed{
\text{Greedy Selection}
+
\text{Optimistic Initialization}
\Rightarrow
\text{Exploratory Behavior}
}
$$

这里没有 $\epsilon$-greedy 的随机探索，但仍然产生了探索行为。

---

## 14. 固定步长（Constant Step Size）

固定步长更新：

$$
Q_{t+1}(a)
=
Q_t(a)
+
\alpha
\left[
R_t-Q_t(a)
\right]
$$

例如：

$$
\alpha=0.1
$$

与样本平均法不同，$\alpha$ 不会随着动作访问次数增加而变小。

代码：

```python
q_values[action] += (
    alpha
    * (reward - q_values[action])
)
```

---

### 14.1 乐观初始值为什么适合配合固定步长观察？

如果使用样本平均：

$$
\alpha=\frac1{N(a)}
$$

第一次访问动作时：

$$
N(a)=1
$$

因此：

$$
Q(a)
\leftarrow
Q(a)+1\cdot[R-Q(a)]
=R
$$

初始乐观值会被第一次奖励完全覆盖。

固定：

$$
\alpha=0.1
$$

则：

$$
Q_{t+1}
=
Q_t+0.1(R_t-Q_t)
$$

乐观估计会逐渐下降，而不是一次消失。

---

## 15. 固定步长手算示例

已知：

$$
Q_0=5,\qquad\alpha=0.1
$$

奖励依次：

$$
R_1=1,\qquad R_2=2,\qquad R_3=0
$$

第一次：

$$
Q_1
=
5+0.1(1-5)
=
4.6
$$

第二次：

$$
Q_2
=
4.6+0.1(2-4.6)
=
4.34
$$

第三次：

$$
Q_3
=
4.34+0.1(0-4.34)
=
3.906
$$

因此：

$$
\boxed{
Q_1=4.6,\quad
Q_2=4.34,\quad
Q_3=3.906
}
$$

---

## 16. $\epsilon$-Greedy 与 Optimistic Greedy 实验

比较：

### 普通 $\epsilon$-Greedy

$$
Q_0(a)=0,\qquad\epsilon=0.1,\qquad\alpha=0.1
$$

### Optimistic Greedy

$$
Q_0(a)=5,\qquad\epsilon=0,\qquad\alpha=0.1
$$

核心实验现象：

1. Optimistic Greedy 早期 Average Reward 较低并产生明显波动；
2. 这是因为高估所有动作迫使 Agent 在早期尝试大量动作；
3. 随着乐观估计逐渐被真实经验修正，Agent 开始更多利用高价值动作；
4. 在当前平稳 Bandit 实验中，后期 Optimistic Greedy 可以获得较高的 Average Reward 和 Optimal Action Rate；
5. 固定 $\epsilon$ 的 $\epsilon$-greedy 即使已经学到较好的动作，仍会持续随机探索，因此持续承担一定探索成本。

---

## 17. 并列最大值处理（Tie-Breaking）

NumPy：

```python
np.argmax(q_values)
```

在多个元素并列最大时，会返回第一个最大值的位置。

例如：

```python
q_values = np.array([5.0, 5.0, 5.0])
```

则：

```python
np.argmax(q_values)
```

返回：

```text
0
```

因此在所有 Agent 使用相同初始化时，早期动作选择可能出现一定同步现象。

更规范的实现可以在所有最大值动作之间随机选择，这称为：

**并列动作处理（Tie-Breaking）**。

---

## 18. 平稳问题与非平稳问题

### 18.1 平稳问题（Stationary Problem）

“平稳（Stationary）”表示环境的奖励分布不随时间变化。

对于当前 Bandit：

$$
q_*(a)=\text{constant}
$$

例如 action 6 从实验开始到结束都保持同一个真实动作价值。

在这种环境中：

- 早期充分探索；
- 找到较好的动作；
- 后期持续利用；

通常是合理的。

---

### 18.2 非平稳问题（Non-stationary Problem）

“非平稳（Non-stationary）”表示环境本身会随时间变化。

此时应该写成：

$$
q_{*,t}(a)
$$

强调真实动作价值与时间 $t$ 有关。

例如：

```text
早期：
action 6 最优

一段时间后：
action 3 变成最优
```

此时 Agent 不仅需要“学习”，还需要持续适应环境变化。

---

## 19. 为什么样本平均法不适合快速适应非平稳环境？

样本平均更新：

$$
Q_{n+1}
=
Q_n
+
\frac1n
(R_n-Q_n)
$$

随着：

$$
n\rightarrow\infty
$$

步长：

$$
\frac1n\rightarrow0
$$

例如：

$$
n=10000
$$

则：

$$
\frac1n=0.0001
$$

此时新的奖励对估计值影响很小。

因此如果环境已经改变，大量旧数据仍然对当前估计产生很强影响。

---

## 20. 固定步长与非平稳环境

固定：

$$
\alpha=0.1
$$

无论已经学习多少次：

$$
Q_{t+1}
=
Q_t
+
0.1(R_t-Q_t)
$$

最新奖励始终具有明显影响。

因此固定步长具有一种重要效果：

> 更重视最近经验，并逐渐降低旧经验的影响。

这使它比简单样本平均更适合非平稳环境。

---

## 21. 当前已经建立的核心更新思想

目前最值得记住的统一形式：

$$
\boxed{
\text{新估计}
=
\text{旧估计}
+
\text{步长}
\times
(\text{目标}-\text{旧估计})
}
$$

在 Bandit 中：

$$
Q_{t+1}(a)
=
Q_t(a)
+
\alpha
[R_t-Q_t(a)]
$$

其中：

$$
R_t-Q_t(a)
$$

表示“实际观察结果”和“当前估计”之间的误差。

后续强化学习中，这种“旧估计 + 步长 × 误差”的结构会反复出现。

---

## 22. 当前代码与数学公式对应关系

### 真实动作价值

数学：

$$
q_*(a)
$$

代码：

```python
true_values[action]
```

### 动作价值估计

数学：

$$
Q(a)
$$

代码：

```python
q_values[action]
```

### 动作选择次数

数学：

$$
N(a)
$$

代码：

```python
action_counts[action]
```

### 最优动作

数学：

$$
a_*=\arg\max_a q_*(a)
$$

代码：

```python
best_action = np.argmax(true_values)
```

### Greedy 动作

数学：

$$
A_t=\arg\max_a Q_t(a)
$$

代码：

```python
action = np.argmax(q_values)
```

### 样本平均更新

数学：

$$
Q(a)
\leftarrow
Q(a)
+
\frac1{N(a)}
[R-Q(a)]
$$

代码：

```python
q_values[action] += (
    reward - q_values[action]
) / action_counts[action]
```

### 固定步长更新

数学：

$$
Q(a)
\leftarrow
Q(a)
+
\alpha[R-Q(a)]
$$

代码：

```python
q_values[action] += (
    alpha * (reward - q_values[action])
)
```

---

## 23. 当前阶段需要牢固掌握的概念

- 多臂老虎机（Multi-Armed Bandit, MAB）
- 智能体（Agent）
- 环境（Environment）
- 动作（Action）
- 奖励（Reward）
- 真实动作价值（True Action Value）
- 动作价值估计（Action-Value Estimate）
- 最优动作（Optimal Action）
- 探索（Exploration）
- 利用（Exploitation）
- $\epsilon$-贪心策略（$\epsilon$-Greedy Policy）
- 样本平均法（Sample-Average Method）
- 步长（Step Size）
- 固定步长（Constant Step Size）
- 随机种子（Random Seed）
- 可复现性（Reproducibility）
- 单次实验（Single Run）
- 多次独立实验（Multiple Independent Runs）
- 平均奖励（Average Reward）
- 最优动作选择比例（Optimal Action Rate）
- 移动平均（Moving Average）
- 乐观初始值（Optimistic Initial Values）
- 并列动作处理（Tie-Breaking）
- 平稳问题（Stationary Problem）
- 非平稳问题（Non-stationary Problem）

---

## 24. 下一阶段

下一步继续学习：

1. 平稳问题（Stationary Problem）与非平稳问题（Non-stationary Problem）的正式建模；
2. 推导固定步长更新的指数加权性质；
3. 理解为什么固定步长会“遗忘”旧数据；
4. 实现非平稳 Bandit；
5. 实验比较：
   - 样本平均法（Sample Average）
   - 固定步长法（Constant Step Size）
6. 之后继续学习置信上界（Upper Confidence Bound, UCB）。
