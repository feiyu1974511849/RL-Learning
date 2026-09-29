# 强化学习（RL）学习路线

> 研究生研究导向版 · 理论 × 代码 × 论文 × RL+VLA × 形式化验证

**适用背景：** 计算机专业研究生；本科软件工程；有 C/C++/Java 基础，Python 与 PyTorch 尚未系统学习；数学按需补充。

**核心目标：** 本学期系统建立 RL 基础，在理解理论的前提下大量写代码；掌握 DQN、Policy Gradient、Actor-Critic、PPO、SAC 等关键算法；具备阅读与复现 RL 论文的能力，并尽快转入 RL + VLA + Formal Verification 的研究实践。

## 学习原则

- 不单独重学高数、概率论、线性代数：遇到 RL 所需数学时现场补齐。
- 不先学完整 Python/PyTorch 再学 RL：用 RL 项目反向驱动 Python 与 PyTorch 上手。
- 每学完一个可实现的算法，立即做小实战；重要算法增加参数实验、对照实验和论文阅读。
- 前期教材主导，中期教材与论文并行，后期以论文和研究问题主导。
- 核心算法尽量自己实现；AI 可解释、Review、Debug，但不以“复制 AI 生成的完整代码”为主要学习方式。

---

## 1. 教材与资料体系

采用“主教材 + 中文辅助 + 理论经典 + 原始论文”的四层结构。不要把多本教材从头到尾重复读一遍，而是围绕当前知识点交叉使用。

| 层级 | 资料 | 用途 |
|---|---|---|
| 主教材 | 《动手学强化学习》 | 课程主线；理论与 Python/PyTorch 实践结合，优先精读。 |
| 中文辅助 | Datawhale EasyRL（蘑菇书） | 换一种中文解释理解概念；快速复习；补充现代算法。 |
| 理论参考 | Sutton & Barto《强化学习（第2版）》中文版 | MDP、MC、TD、SARSA、Q-learning 等经典理论的深入参考。 |
| 研究阶段 | DQN / PPO / SAC / Safe RL 等原始论文 | 训练论文阅读、公式到代码映射、复现实验和研究能力。 |

**使用方法：** 每个单元由 GPT 先系统讲解；再阅读主教材对应章节；概念不清时看蘑菇书；经典 RL 需要理论加深时查 Sutton；进入 DQN、PPO、SAC 后加入原始论文。

---

## 2. 每个知识单元的固定学习闭环

以后使用 GPT 学习时，建议每一小节都要求按照下面的顺序进行：

1. 问题背景
2. 直觉解释
3. 数学定义 / 推导
4. 算法流程与伪代码
5. 最小代码实现
6. Gymnasium 实战
7. 训练曲线与结果分析
8. 参数实验 / Debug
9. 小练习
10. 教材与论文回顾

**代码训练分级：**

- 前期：补全关键代码。
- 中期：根据伪代码独立实现。
- 后期：根据论文公式和算法描述自行设计实现。

---

## 3. 总体知识路线

```text
Python/NumPy
→ RL 基本概念
→ MDP
→ Value/Q/Advantage
→ Bellman
→ DP
→ Monte Carlo
→ TD
→ SARSA/Q-learning
→ PyTorch/函数逼近
→ DQN
→ Policy Gradient
→ Actor-Critic/GAE
→ PPO
→ 连续控制（DDPG/TD3/SAC）
→ RL+VLA
→ Safe RL / Formal Verification
→ 论文研究
```

### 需要形成研究级理解的主干

```text
MDP → Bellman → TD → Q-learning → DQN → Policy Gradient → Actor-Critic → PPO → SAC
```

DP、MC、SARSA、DDPG、TD3 等仍要学习，但投入程度可以低于主干算法。PPO/SAC 之后不要无限刷算法，应尽快转入研究问题。

---

## 4. 16 周建议进度

| 周次 | 理论主题 | 核心内容 | 代码 / 产出 | 要求 |
|---|---|---|---|---|
| 0-1 | Python for RL + NumPy | Python 基本语法、函数/类、NumPy、Matplotlib；Gymnasium 基础 | 多臂老虎机 / 简单环境；能读写基本 Python | 不单独学完整 Python 课程 |
| 2 | RL 基础 + MDP | Agent/Environment、State/Observation、Action、Reward、Return、Policy、Trajectory、Markov Property | 手写极简 GridWorld；观察 trajectory | 重点区分 State 与 Observation |
| 3 | Value + Bellman + DP | Vπ、Qπ、Bellman expectation/optimality；Policy Evaluation/Iteration、Value Iteration | GridWorld 求 V 与最优策略 | Bellman 必须会从 Return 理解 |
| 4 | MC + TD | MC Prediction/Control、TD(0)、Bootstrapping、TD Error | Random Walk / Blackjack 小实验 | 理解 MC 与 TD 差异 |
| 5 | SARSA + Q-learning | On-policy / Off-policy、ε-greedy、TD Control | CliffWalking：SARSA vs Q-learning | 做 ε、α、γ 参数实验 |
| 6 | PyTorch for RL | Tensor、nn.Module、forward、autograd、optimizer、detach/no_grad | 用小网络拟合函数 / Q 值 | 只学 DQN 所需 PyTorch |
| 7 | DQN | 函数逼近、Replay Buffer、Target Network、TD Target | 从零实现 CartPole DQN | 开始论文式实验记录 |
| 8 | DQN 论文 + 改进 | DQN 原论文；Double DQN；了解 Dueling/PER | DQN vs Double DQN | 完成第一次论文 → 代码映射 |
| 9 | Policy Gradient | 随机策略、目标函数、log trick、REINFORCE、baseline | CartPole REINFORCE | 数学按需补概率分布 / 梯度 |
| 10 | Actor-Critic + Advantage | Actor/Critic、TD Error、Advantage、A2C 思想 | 手写 Actor-Critic | 建立 value-based 与 policy-based 联系 |
| 11 | GAE + PPO | Importance Sampling、GAE、ratio、clip objective、entropy | 手写 PPO（核心部分必须自己写） | 本学期重点之一 |
| 12 | PPO 论文与实验 | PPO 原论文；公式-代码-实验对应 | PPO 参数 / 消融小实验 | 达到能解释每一项 loss |
| 13 | 连续控制 | DDPG、TD3 思想；确定性策略；连续动作 | Pendulum 等连续环境 | DDPG/TD3 以理解演进为主 |
| 14 | SAC | Maximum Entropy RL、随机策略、reparameterization、温度参数 | SAC 连续控制实验 | 本学期重点之一 |
| 15 | RL + VLA | VLA 作为 policy/基础模型；RL fine-tuning；reward；online/offline；安全约束接口 | 调用现有 VLA，搭建最小 RL/VLA 实验链路 | 不深入 VLA 架构细节 |
| 16+ | RL + Formal Verification | Safe RL、CMDP、Shielding、Temporal Logic、Model Checking、Runtime/Policy Verification | 围绕 VLA/RL 加入安全约束或验证原型 | 从课程模式转入 Research Mode |

---

## 5. 分阶段详细 Syllabus

### 阶段 A：Python 与 RL 最小预备

- **目标：** 能独立读写后续 RL 示例，而不是成为 Python 语言专家。
- **Python：** 变量、容器、循环、函数、类、模块；重点熟悉 Python 与 C/C++/Java 不同的写法。
- **NumPy：** ndarray、shape、索引/切片、广播、argmax、mean、随机采样、向量化。
- **Matplotlib：** 能画 episode-return、loss、moving average 曲线。
- **Gymnasium：** reset、step、observation、reward、terminated、truncated、action_space。

**验收：** 能不依赖 AI 完整代写，自己修改一个简单 Agent 的动作选择和训练循环。

### 阶段 B：RL 基础与表格型强化学习

- **基本概念：** Agent、Environment、State、Observation、Action、Reward、Episode、Trajectory、Policy、Return、Discount Factor。
- **MDP：** 状态空间、动作空间、转移概率、奖励函数、折扣因子；Markov Property。
- **价值函数：** Vπ(s)、Qπ(s,a)、后续引入 Advantage。
- **Bellman：** Bellman Expectation Equation 与 Bellman Optimality Equation。
- **DP：** Policy Evaluation、Policy Improvement、Policy Iteration、Value Iteration。
- **MC：** 完整回合估计 Return；Prediction/Control。
- **TD：** TD(0)、Bootstrapping、TD Error。
- **Control：** SARSA、Q-learning；On-policy 与 Off-policy。

**小实战链：**

```text
Bandit → GridWorld → Random Walk/Blackjack → CliffWalking/FrozenLake
```

### 阶段 C：Deep RL 与 DQN

- 按需学习 PyTorch：Tensor、Module、Linear/ReLU、forward、loss.backward、optimizer.step、detach、no_grad、device。
- 理解为什么 Q-table 无法处理高维/连续状态，进入函数逼近 `Qθ(s,a)`。
- **DQN：** Experience Replay、Replay Buffer、Target Network、TD Target、ε-greedy。
- **Double DQN：** 理解 max 引起的过估计问题；Dueling/PER 以理解思想为主。

**论文节点 1：** Mnih et al., *Human-level control through deep reinforcement learning*。要求能把论文中的 Replay、Target Network、TD target 映射到自己的代码。

### 阶段 D：Policy Gradient → Actor-Critic → PPO

- **REINFORCE：** 从 `J(θ)` 与 trajectory distribution 理解 Policy Gradient，而不是背结论。
- **Baseline / Advantage：** 理解降低方差以及 Q、V、A 的关系。
- **Actor-Critic：** Actor 负责策略，Critic 估计价值；理解 TD Error 如何连接二者。
- **GAE：** 理解 bias-variance trade-off 以及 λ 的意义。
- **PPO：** Importance Sampling、probability ratio、clipped surrogate objective、value loss、entropy bonus。

**论文节点 2：** Schulman et al., *Proximal Policy Optimization Algorithms*。PPO 是本路线的核心算法之一，要求能从论文公式写出核心 PyTorch loss。

### 阶段 E：连续控制与 SAC

- 理解离散动作与连续动作的差异，以及机器人动作空间为何常是连续的。
- **DDPG → TD3：** 重点理解算法演进和 deterministic actor-critic。
- **SAC：** Maximum Entropy RL、stochastic policy、reparameterization trick、temperature α。

**论文节点 3：** SAC 相关经典论文。目标是能在连续控制环境运行、调参并解释 entropy 对探索的作用。

---

## 6. 面向研究方向的后半程

### RL + VLA

VLA 不作为主要研究对象，重点是将现有 VLA/策略模型作为可调用组件，研究 RL 如何介入动作策略优化、奖励设计、安全约束与验证。

```text
Observation
→ VLA/Policy
→ Action
→ Environment
→ Reward/Constraint
→ RL Optimization
```

重点关注：

- policy fine-tuning
- reward design
- online/offline interaction
- continuous control
- safety constraints
- sim-to-real

VLA 架构只补足调用、输入输出、动作表示和训练接口所需知识，不陷入大模型结构细节。

### RL + Formal Verification

从“最大化期望回报”逐步扩展到“在优化性能的同时满足可验证的安全/逻辑约束”。

- Safe Reinforcement Learning（安全强化学习）
- Constrained MDP（约束马尔可夫决策过程）
- Shielding（安全屏蔽 / 安全盾）
- Reachability（可达性分析）
- Temporal Logic（时序逻辑）
- Model Checking（模型检测）
- Runtime Verification（运行时验证）
- Policy / Neural Network Verification（策略 / 神经网络验证）

长期研究框架可以抽象为：

```text
VLA Policy
→ RL Optimization
→ Formal Safety Constraint / Verification / Shield
→ Safe Action
→ Robot/Environment
```

---

## 7. 每个算法的小实战模板

以后每学一个算法，至少完成以下内容；重要算法（DQN/PPO/SAC）增加论文与对照实验。

| 环节 | 要求 |
|---|---|
| 实现 | 先实现算法核心，不直接调用成熟 RL 算法库代替学习。 |
| 环境 | 选择一个足够小、能快速训练的 Gymnasium 环境。 |
| 可视化 | 至少记录 episode return；需要时增加 loss、Q/V、entropy、success rate。 |
| 参数实验 | 改变 1-2 个关键超参数，观察训练稳定性、探索、收敛速度。 |
| 解释 | 写下“现象是什么、为什么可能这样、下一步怎么验证”。 |
| 复现性 | 记录随机种子、环境版本、超参数和依赖版本。 |

---

## 8. 实验记录模板

从 Q-learning 开始养成研究记录习惯。每个实验可以保存一个 `README.md` / `experiment.md`。

```markdown
# 实验名称

## 研究问题 / 假设

## 算法与环境

## 关键超参数
- α：
- γ：
- ε：
- learning rate：
- batch size：
- seed：

## 评价指标
- Return
- Success Rate
- Constraint Violation
- Loss

## 结果

## 异常现象

## 我的解释

## 下一步实验

## 结论
```

尤其进入 RL+VLA+形式化验证后，应增加：

- constraint violation rate
- safety success rate
- verification result

等安全相关指标。

---

## 9. 论文阅读与复现能力训练

论文不是单独的一门课，而是在算法基础建立后自然插入。建议按照以下顺序阅读：

| 节点 | 论文 / 材料 | 阅读目标 |
|---|---|---|
| DQN | Mnih et al. - *Human-level control through deep reinforcement learning* | Replay、Target Network、深度 Q-learning |
| Double DQN | van Hasselt et al. - *Deep Reinforcement Learning with Double Q-learning* | Q-value overestimation |
| Policy Gradient | 经典 Policy Gradient / REINFORCE 相关材料 | 策略梯度推导与 stochastic policy |
| PPO | Schulman et al. - *Proximal Policy Optimization Algorithms* | ratio、clip、advantage、稳定更新 |
| SAC | Haarnoja et al. - *Soft Actor-Critic* 系列 | Maximum Entropy、连续控制 |
| 研究阶段 | Safe RL / RL verification / VLA-RL 方向论文 | 从综述与近年论文建立自己的 related work 图谱 |

**阅读方式：**

```text
Abstract
→ Introduction
→ Problem Formulation
→ Method/Algorithm
→ 核心公式
→ Experiments
→ Ablation
→ 与自己代码逐项映射
```

初期由 GPT 带读，后期先自己读再讨论。

---

## 10. 阶段验收标准

| 等级 | 能力 |
|---|---|
| Level 1 | 能准确解释 RL 核心概念、MDP、Bellman、TD、on/off-policy。 |
| Level 2 | 能根据公式/伪代码自己实现 Q-learning、DQN、REINFORCE/Actor-Critic、PPO 的核心部分。 |
| Level 3 | 能阅读 RL 论文，理解核心公式与实验，并完成小规模复现或改进。 |
| Level 4 | 能从 RL+VLA+Formal Verification 的现有工作中发现限制，提出研究问题并设计可验证实验。 |

**本学期目标：** 至少稳定达到 Level 3，并开始向 Level 4 过渡。

---

## 11. 推荐项目目录

```text
rl-learning/
├── 00_python_for_rl/
├── 01_bandit/
├── 02_gridworld_mdp/
├── 03_dynamic_programming/
├── 04_monte_carlo_td/
├── 05_sarsa_qlearning/
├── 06_pytorch_for_rl/
├── 07_dqn/
├── 08_double_dqn/
├── 09_reinforce/
├── 10_actor_critic/
├── 11_ppo/
├── 12_sac/
├── 13_vla_rl/
├── 14_safe_rl_verification/
└── experiments/
```

---

## 12. 以后如何用 GPT 按这份路线学习

每次开始新小节时，可以直接给 GPT 类似下面的指令：

> 按照我的 RL 学习路线，现在学习【Q-learning】。先检查它依赖的前置知识；用中文从问题动机、直觉、数学推导、伪代码讲起。必要数学现场补。理论讲完后带我做一个小实战，但不要一开始把完整答案代码全部给我，优先让我补全关键部分或根据伪代码自己写。最后做参数实验、结果分析和小结，并告诉我对应教材/论文阅读任务。

进入论文阶段可以改为：

> 我现在已经学完 PPO 基础。带我读 PPO 原论文。不要只总结论文；先让我理解 problem formulation 和核心公式，再把公式逐项映射到 PyTorch 实现，最后设计一个可复现的小实验和消融实验。

---

## 13. 路线使用说明

- 进度表不是死规定。如果某一算法理解不牢，优先把理论和实验做扎实，而不是追赶周数。
- Python/PyTorch 的目标是服务 RL 研究；遇到语言问题即时补，不把语言学习变成新的主线。
- DQN、PPO、SAC 是重要里程碑；PPO 和连续控制尤其与后续 VLA/机器人策略研究相关。
- 进入研究阶段后，路线应根据导师课题、所用 VLA、仿真平台和形式化验证方法继续迭代。
- 真正的终点不是“学完所有 RL 算法”，而是能够独立阅读文献、实现方法、设计实验、定位问题，并形成自己的研究问题。

---

**版本：** RL 学习路线 v1.0 · 研究生研究导向
