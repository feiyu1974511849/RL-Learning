# 第 0 章 Python for RL

## 0.1 Python 核心语法

### 1. 变量与动态类型

Python 是**动态类型语言（Dynamically Typed Language）**，定义变量时不需要显式声明类型。

```python
x = 10
reward = 1.5
name = "DQN"
```

与 C/C++、Java 不同，不需要：

```cpp
int x = 10;
double reward = 1.5;
```

---

### 2. 缩进与代码块

Python 不使用 `{}` 表示代码块，而是使用**缩进**。

通常使用 4 个空格。

```python
x = 10

if x > 5:
    print("A")
    print("B")

print("C")
```

其中：

```python
print("A")
print("B")
```

属于 `if` 代码块。

而：

```python
print("C")
```

不属于 `if`。

---

### 3. List：列表

List（列表）是一种可以保存多个元素的数据结构。

```python
rewards = [1, 2, 3, 4]
```

访问元素：

```python
rewards[0]    # 1
rewards[2]    # 3
```

Python 索引从 `0` 开始。

添加元素：

```python
rewards.append(5)
```

此时：

```python
rewards
```

为：

```text
[1, 2, 3, 4, 5]
```

RL 中经常使用列表保存训练数据：

```python
episode_rewards = []

episode_rewards.append(reward)
```

---

### 4. 切片 Slice

切片（Slice）用于从序列中取出一部分元素。

```python
a = [10, 20, 30, 40, 50]
```

基本形式：

```python
a[start:end]
```

范围为：

```text
[start, end)
```

即**左闭右开**。

例如：

```python
a[1:4]
```

结果：

```text
[20, 30, 40]
```

省略起点：

```python
a[:3]
```

结果：

```text
[10, 20, 30]
```

省略终点：

```python
a[2:]
```

结果：

```text
[30, 40, 50]
```

负数索引：

```python
a[-1]
```

表示最后一个元素：

```text
50
```

RL 中常见：

```python
rewards[-100:]
```

表示最后 100 个 Reward。

---

### 5. Tuple：元组

Tuple（元组）通常用于保存一组相关数据。

```python
transition = (state, action, reward, next_state)
```

RL 中一次状态转移（Transition）通常可以表示为：

```text
(sₜ, aₜ, rₜ₊₁, sₜ₊₁)
```

例如：

```python
transition = (3, 1, 10, 4)
```

#### 元组拆包 Tuple Unpacking

```python
state, action, reward, next_state = transition
```

得到：

```text
state = 3
action = 1
reward = 10
next_state = 4
```

Gymnasium 中常见：

```python
state, info = env.reset()
```

以及：

```python
next_state, reward, terminated, truncated, info = env.step(action)
```

---

### 6. Dict：字典

Dict（Dictionary，字典）保存键值对（Key-Value Pair）。

类似于：

* C++ 的 `std::unordered_map`
* Java 的 `HashMap`

例如：

```python
config = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 0.1
}
```

访问：

```python
config["gamma"]
```

得到：

```text
0.99
```

RL 中经常用于保存：

* 超参数
* 配置信息
* 日志
* 环境信息

---

### 7. `for` 循环与 `range()`

基本形式：

```python
for i in range(10):
    print(i)
```

`range(10)` 对应：

```text
0, 1, 2, ..., 9
```

即：

```text
[0, 10)
```

RL 中经常使用：

```python
for episode in range(1000):
    ...
```

表示训练 1000 个 Episode（回合）。

---

### 8. 直接遍历 List

```python
rewards = [10, 20, 30]

for reward in rewards:
    print(reward)
```

如果同时需要索引和值，可以使用：

```python
for i, reward in enumerate(rewards):
    print(i, reward)
```

输出：

```text
0 10
1 20
2 30
```

---

### 9. `enumerate()`

`enumerate()` 可以同时获得元素的索引和值。

```python
for index, value in enumerate(data):
    ...
```

例如：

```python
actions = ["left", "right", "up"]

for i, action in enumerate(actions):
    print(i, action)
```

输出：

```text
0 left
1 right
2 up
```

---

### 10. 不需要循环变量时使用 `_`

如果只需要循环一定次数，而不使用循环变量：

```python
for _ in range(10000):
    ...
```

`_` 是 Python 中常见的惯例，表示：

> 这个变量不会被使用。

---

### 11. 函数

定义函数：

```python
def add(a, b):
    return a + b
```

调用：

```python
result = add(1, 2)
```

#### 默认参数

```python
def train(episodes=1000, gamma=0.99):
    ...
```

可以：

```python
train()
```

也可以：

```python
train(episodes=5000)
```

或者：

```python
train(
    episodes=5000,
    gamma=0.95
)
```

---

### 12. 多返回值

Python 函数可以返回多个值：

```python
def test():
    return 10, 20
```

接收：

```python
a, b = test()
```

本质上相当于返回一个 Tuple，然后进行元组拆包。

Gymnasium 中大量使用这种形式：

```python
state, info = env.reset()
```

---

### 13. Class：类

基本形式：

```python
class Agent:

    def __init__(self, name):
        self.name = name

    def act(self):
        print(self.name)
```

创建对象：

```python
agent = Agent("DQN")
```

调用方法：

```python
agent.act()
```

#### `__init__`

`__init__()` 用于初始化对象，可以近似理解为构造函数。

```python
def __init__(self):
    self.total_reward = 0
```

#### `self`

`self` 表示当前对象，类似于 C++ 和 Java 中的 `this`。

例如：

```python
self.name
```

可以理解为 Java 中：

```java
this.name
```

RL 中以后会经常写：

```python
class QLearningAgent:

    def __init__(self):
        self.Q = ...

    def take_action(self, state):
        ...

    def update(self, state, action, reward, next_state):
        ...
```

---

### 14. `import`

导入模块：

```python
import numpy
```

使用别名：

```python
import numpy as np
```

于是：

```python
numpy.array(...)
```

可以简写成：

```python
np.array(...)
```

RL 中常见：

```python
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
```

---

### 15. 条件表达式

Python：

```python
action = 0 if state < 5 else 1
```

等价于 C/C++ 中：

```cpp
int action = state < 5 ? 0 : 1;
```

Python 的形式为：

```python
value_if_true if condition else value_if_false
```

---

### 16. List Comprehension：列表推导式

```python
squares = [x * x for x in range(10)]
```

等价于：

```python
squares = []

for x in range(10):
    squares.append(x * x)
```

例如：

```python
returns = [x / 100 for x in rewards]
```

---

### 17. 对象引用与复制

```python
a = [1, 2, 3]
b = a
```

这里没有复制列表。

`a` 和 `b` 指向同一个对象。

因此：

```python
b[0] = 100
```

之后：

```python
print(a)
```

得到：

```text
[100, 2, 3]
```

如果需要创建一个新的列表：

```python
b = a.copy()
```

需要注意，`copy()` 对嵌套对象执行的是**浅拷贝（Shallow Copy）**。更复杂的深拷贝问题以后遇到时再学习。

---

### 18. f-string

f-string 用于格式化字符串：

```python
name = "DQN"

print(f"Algorithm: {name}")
```

可以控制浮点数显示精度：

```python
probability = 0.9333333

print(f"{probability:.4f}")
```

输出：

```text
0.9333
```

其中：

```text
.4f
```

表示以浮点数形式显示，并保留 4 位小数。

---

## 0.2 NumPy

### 1. 导入 NumPy

```python
import numpy as np
```

NumPy 最核心的数据结构是：

**ndarray（N-dimensional Array，N 维数组）**

创建数组：

```python
a = np.array([1, 2, 3])
```

---

### 2. Python List 与 NumPy Array

Python List：

```python
a = [1, 2, 3]

print(a * 2)
```

结果：

```text
[1, 2, 3, 1, 2, 3]
```

NumPy Array：

```python
a = np.array([1, 2, 3])

print(a * 2)
```

结果：

```text
[2 4 6]
```

NumPy Array 支持方便的数值运算。

---

### 3. 向量逐元素运算

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
```

加法：

```python
a + b
```

结果：

```text
[11 22 33]
```

逐元素乘法：

```python
a * b
```

结果：

```text
[10 40 90]
```

这里的 `*` 表示：

**逐元素乘法（Element-wise Multiplication）**

即：

```text
[1,2,3] * [10,20,30] = [10,40,90]
```

---

### 4. `@`：矩阵乘法运算符

NumPy 中：

```python
a @ b
```

使用 `@` 运算符执行矩阵乘法相关运算。

对于两个一维数组：

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
```

执行：

```python
result = a @ b
```

NumPy 会计算两个一维数组的**内积（Dot Product）**：

```text
1 × 10 + 2 × 20 + 3 × 30 = 140
```

因此：

```python
print(a @ b)
```

输出：

```text
140
```

对于二维数组，`@` 可以执行通常意义上的矩阵乘法。

例如：

```python
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

C = A @ B
```

---

### 5. `shape`

`shape` 表示数组各个维度的大小。

一维数组：

```python
a = np.array([1, 2, 3])

print(a.shape)
```

结果：

```text
(3,)
```

表示这是一个长度为 3 的一维数组。

二维数组：

```python
Q = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
```

```python
print(Q.shape)
```

结果：

```text
(2, 3)
```

表示一个：

```text
2 × 3
```

的二维数组。

对于二维数组：

```python
Q.shape[0]
```

表示行数。

```python
Q.shape[1]
```

表示列数。

如果 Q-table 满足：

```text
Q.shape = (num_states, num_actions)
```

那么：

```python
Q.shape[0]
```

表示状态数量。

```python
Q.shape[1]
```

表示动作数量。

---

### 6. 创建全零数组

```python
np.zeros(5)
```

得到：

```text
[0. 0. 0. 0. 0.]
```

创建二维数组：

```python
Q = np.zeros((5, 3))
```

可以用于初始化 Q-table。

例如初始时令：

```text
Q(s,a) = 0
```

其中：

```text
5 = State 数量
3 = Action 数量
```

---

### 7. 创建全一数组

```python
np.ones(5)
```

得到：

```text
[1. 1. 1. 1. 1.]
```

---

### 8. NumPy 索引

例如：

```python
Q = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
```

访问一个元素：

```python
Q[1, 2]
```

结果：

```text
60
```

访问一整行：

```python
Q[1]
```

结果：

```text
[40 50 60]
```

RL 中：

```python
Q[state]
```

表示：

> 当前状态 `state` 下所有动作对应的 Q-value。

而：

```python
Q[state, action]
```

对应：

```text
Q(s,a)
```

---

### 9. Q-table

Q-table（Q 表）可以使用二维 NumPy 数组表示。

例如：

```python
Q = np.zeros((5, 3))
```

结构可以理解为：

```text
             Action
          0    1    2
       ┌──────────────
s=0    │ 0    0    0
s=1    │ 0    0    0
s=2    │ 0    0    0
s=3    │ 0    0    0
s=4    │ 0    0    0
```

其中：

```python
Q[s, a]
```

表示：

```text
Q(s,a)
```

---

### 10. `np.max()`

`np.max()` 返回最大值。

```python
values = np.array([3.2, 7.5, 1.8])

np.max(values)
```

结果：

```text
7.5
```

RL 中：

```python
np.max(Q[state])
```

对应：

```text
maxₐ Q(s,a)
```

表示：

> 当前状态下所有动作中最大的 Q-value。

---

### 11. `np.argmax()`

`np.argmax()` 返回最大值所在的索引。

```python
values = np.array([3.2, 7.5, 1.8])

np.argmax(values)
```

结果：

```text
1
```

RL 中：

```python
action = np.argmax(Q[state])
```

对应：

```text
a* = arg maxₐ Q(s,a)
```

表示：

> 找到使 Q-value 最大的动作。

---

### 12. `max` 与 `argmax` 的区别

假设：

```python
Q[state]
```

为：

```text
[3.2, 7.5, 1.8]
```

那么：

```python
np.max(Q[state])
```

结果：

```text
7.5
```

对应：

```text
maxₐ Q(s,a) = 7.5
```

而：

```python
np.argmax(Q[state])
```

结果：

```text
1
```

对应：

```text
arg maxₐ Q(s,a) = 1
```

因此：

* `max`：最大的**值**
* `argmax`：最大值对应的**位置/动作**

---

### 13. `np.mean()`

计算平均值：

```python
rewards = np.array([10, 20, 15, 30, 25])

np.mean(rewards)
```

结果：

```text
20
```

数学上：

```text
R̄ = (1/N) Σᵢ₌₁ᴺ Rᵢ
```

RL 中常见：

```python
np.mean(rewards[-100:])
```

表示：

> 最近 100 个 Episode 的平均 Reward。

---

### 14. NumPy 随机数

生成 `[0, 1)` 范围内的随机浮点数：

```python
np.random.random()
```

例如可能得到：

```text
0.3728
```

RL 中可以用于概率判断：

```python
if np.random.random() < epsilon:
    ...
```

---

### 15. `np.random.randint()`

随机生成整数。

```python
np.random.randint(4)
```

可能产生：

```text
0
1
2
3
```

不会产生 `4`。

RL 中可以用于随机选择动作：

```python
action = np.random.randint(num_actions)
```

---

### 16. `np.random.choice()`

随机选择一个元素：

```python
np.random.choice([0, 1, 2])
```

也可以按照指定概率选择：

```python
np.random.choice(
    [0, 1, 2],
    p=[0.1, 0.2, 0.7]
)
```

对应：

```text
P(a=0) = 0.1
```

```text
P(a=1) = 0.2
```

```text
P(a=2) = 0.7
```

后续可以用于理解：

```text
a ~ π(a | s)
```

即：

> 根据策略给出的动作概率分布采样一个动作。

---

### 17. `axis`

例如：

```python
Q = np.array([
    [1, 5, 3],
    [7, 2, 4]
])
```

整体最大值：

```python
np.max(Q)
```

结果：

```text
7
```

每行分别求最大值：

```python
np.max(Q, axis=1)
```

结果：

```text
[5 7]
```

即：

```text
[1, 5, 3] → 5
[7, 2, 4] → 7
```

如果：

```text
Q.shape = (batch_size, num_actions)
```

那么沿动作维度取最大值，可以理解为：

> 对 Batch 中的每个 State，分别寻找最大的 Q-value。

---

### 18. Broadcasting：广播机制

Broadcasting（广播机制）允许不同形状的数据在满足一定规则时直接进行运算。

例如：

```python
a = np.array([1, 2, 3])

result = a + 10
```

结果：

```text
[11 12 13]
```

可以直观理解为 NumPy 自动进行了：

```text
[1,2,3] + 10 = [1+10,2+10,3+10]
```

所以：

```text
[1,2,3] + 10 = [11,12,13]
```

PyTorch 中也大量存在 Broadcasting。

---

## 0.3 已接触的强化学习基础概念

### 1. Agent

Agent（智能体）是强化学习中负责：

* 观察环境
* 选择动作
* 接收奖励
* 学习策略

的主体。

代码中通常会封装成类：

```python
class Agent:

    def __init__(self):
        ...

    def take_action(self, state):
        ...

    def update(self, ...):
        ...
```

---

### 2. Environment

Environment（环境）是 Agent 进行交互的对象。

Agent 向环境执行 Action，环境返回新的状态和 Reward。

基本交互过程可以表示为：

```text
Agent
  │
  │ Action
  ▼
Environment
  │
  │ State / Reward
  ▼
Agent
```

---

### 3. State

State（状态）表示 Agent 当前所处的状态。

通常记为：

```text
s
```

或者：

```text
sₜ
```

例如：

```python
state = 2
```

---

### 4. Action

Action（动作）表示 Agent 可以执行的行为。

通常记为：

```text
a
```

或者：

```text
aₜ
```

例如环境有三个动作：

```text
0
1
2
```

---

### 5. Reward

Reward（奖励）是环境反馈给 Agent 的数值信号。

通常记为：

```text
r
```

或者：

```text
rₜ
```

例如：

```python
reward = 1
```

---

### 6. Episode

Episode（回合）表示 Agent 从一次交互开始到该次交互过程结束所形成的完整过程。

训练代码中经常看到：

```python
for episode in range(1000):
    ...
```

表示进行 1000 个 Episode。

---

### 7. Transition

Transition（状态转移）表示一次环境交互。

常见形式：

```text
(sₜ, aₜ, rₜ₊₁, sₜ₊₁)
```

代码中可以使用 Tuple：

```python
transition = (
    state,
    action,
    reward,
    next_state
)
```

---

### 8. Q-value

Q-value（动作价值）记为：

```text
Q(s,a)
```

目前可以先直观理解为：

> 在状态 `s` 下执行动作 `a` 的价值。

表格型强化学习中可以使用二维数组保存：

```python
Q[state, action]
```

因此：

```text
Q(s,a)
```

对应：

```python
Q[state, action]
```

Q-value 的严格定义将在后续正式学习价值函数时给出。

---

### 9. 贪心动作

如果当前状态为 `s`，选择 Q-value 最大的动作：

```text
a* = arg maxₐ Q(s,a)
```

代码：

```python
action = np.argmax(Q[state])
```

---

### 10. Exploration vs. Exploitation

**探索与利用（Exploration vs. Exploitation）**是强化学习中的核心问题之一。

#### Exploration：探索

尝试当前不一定最优的动作，以获取更多信息。

例如：

```python
action = np.random.randint(num_actions)
```

#### Exploitation：利用

利用当前已经获得的信息，选择当前认为最好的动作：

```python
action = np.argmax(Q[state])
```

---

### 11. ε-greedy：ε-贪心策略

ε-greedy（ε-贪心策略）的基本思想：

```text
a =
    随机动作，概率为 ε
    arg maxₐ Q(s,a)，概率为 1 - ε
```

代码：

```python
if np.random.random() < epsilon:
    action = np.random.randint(num_actions)
else:
    action = np.argmax(Q[state])
```

其中：

* `epsilon`：探索概率
* `1 - epsilon`：进入贪心利用分支的概率

---

### 12. ε-greedy 的实际动作概率

假设有 3 个动作：

```text
0
1
2
```

并且：

```text
ε = 0.1
```

当前唯一的贪心动作是：

```text
a* = 2
```

探索时三个动作均匀随机选择。

对于非贪心动作：

```text
P(a=0) = ε × (1/3)
```

代入：

```text
P(a=0) = 0.1 × (1/3) ≈ 0.0333
```

同理：

```text
P(a=1) ≈ 0.0333
```

对于贪心动作 `2`，它有两种被选中的情况：

1. 进入利用分支；
2. 进入探索分支，但随机时恰好选中动作 `2`。

因此：

```text
P(a=2) = (1 - ε) + ε × (1/3)
```

代入：

```text
P(a=2) = 0.9 + 0.1 × (1/3)
```

所以：

```text
P(a=2) ≈ 0.9333
```

因此，严格来说：

> `epsilon = 0.1` 表示有 10% 的概率进入随机探索分支，而不是“最优动作只有 90% 的概率被选择”。

因为随机探索时，也有可能随机选中贪心动作。

---

## 0.4 数学公式与代码对应关系

### 1. Q-value

数学：

```text
Q(s,a)
```

代码：

```python
Q[state, action]
```

---

### 2. 当前状态所有动作的 Q-value

对于状态 `s` 的所有动作价值，可以通过：

```python
Q[state]
```

获得。

例如：

```python
Q[state]
```

得到：

```text
[3.2, 7.5, 1.8]
```

表示当前状态下三个动作的 Q-value。

---

### 3. 最大 Q-value

数学：

```text
maxₐ Q(s,a)
```

代码：

```python
np.max(Q[state])
```

---

### 4. 贪心动作

数学：

```text
arg maxₐ Q(s,a)
```

代码：

```python
np.argmax(Q[state])
```

---

### 5. 最近 100 个 Episode 的平均 Reward

代码：

```python
np.mean(rewards[-100:])
```

其中：

```python
rewards[-100:]
```

取最后 100 个 Reward。

然后：

```python
np.mean(...)
```

计算平均值。

---

### 6. ε-greedy

数学：

```text
a =
    random action，概率为 ε
    arg maxₐ Q(s,a)，概率为 1 - ε
```

代码：

```python
if np.random.random() < epsilon:
    action = np.random.randint(num_actions)
else:
    action = np.argmax(Q[state])
```

---

## 0.5 常见代码速查

### Python

```python
a = [1, 2, 3]
```

```python
a.append(4)
```

```python
a[-1]
```

```python
a[-100:]
```

```python
for x in data:
    ...
```

```python
for i, x in enumerate(data):
    ...
```

```python
for _ in range(10000):
    ...
```

```python
def function_name(parameter):
    return value
```

```python
class Agent:

    def __init__(self):
        self.value = 0
```

---

### NumPy

```python
import numpy as np
```

```python
a = np.array([1, 2, 3])
```

```python
Q = np.zeros((num_states, num_actions))
```

```python
Q.shape
```

```python
Q.shape[0]
```

```python
Q.shape[1]
```

```python
Q[state]
```

```python
Q[state, action]
```

```python
np.max(Q[state])
```

```python
np.argmax(Q[state])
```

```python
np.mean(rewards)
```

```python
np.mean(rewards[-100:])
```

```python
np.random.random()
```

```python
np.random.randint(num_actions)
```

```python
np.random.choice(actions, p=probabilities)
```

---

### ε-greedy

```python
def epsilon_greedy(Q, state, epsilon):

    if np.random.random() < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state])

    return action
```
