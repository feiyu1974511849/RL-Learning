
"""
文件名称：02_autograd.py

实验主题：PyTorch 自动求导机制（Autograd）

实验内容：
1. requires_grad 与计算图
2. 单变量和多变量自动求导
3. 梯度累积与清零
4. detach() 与 torch.no_grad()
5. 使用自动求导验证线性层的矩阵梯度

前置知识：
- 偏导数与链式法则
- 梯度下降法
- 张量形状与矩阵乘法

运行方式：
python 03_deep_learning_basics/02_autograd.py
"""

import torch


def experiment_1_requires_grad():
    """实验1：追踪计算图并查看叶子张量的梯度。"""

    print("\n========== 实验1：requires_grad ==========")

    # 输入数据与真实值不需要梯度。
    x = torch.tensor(2.0)
    y = torch.tensor(10.0)

    # w 和 b 是待训练参数，需要计算梯度。
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # 前向传播：预测值 = wx + b。
    y_pred = w * x + b

    # 平方误差损失。
    loss = (y_pred - y) ** 2

    print("预测值：", y_pred.item())
    print("损失值：", loss.item())

    # 检查计算图相关属性。
    print("w.requires_grad：", w.requires_grad)
    print("w.grad_fn：", w.grad_fn)
    print("y_pred.grad_fn：", y_pred.grad_fn)
    print("loss.grad_fn：", loss.grad_fn)

    # 执行反向传播。
    # 自动计算损失对 w、b 的偏导数。
    loss.backward()

    print("w 的梯度：", w.grad.item())
    print("b 的梯度：", b.grad.item())

    # 手算结果：
    # y_pred = 1*2+0 = 2
    # dL/dw = 2*(2-10)*2 = -32
    # dL/db = 2*(2-10) = -16
    assert torch.allclose(w.grad, torch.tensor(-32.0))
    assert torch.allclose(b.grad, torch.tensor(-16.0))


def experiment_2_multivariable():
    """实验2：多参数模型的自动求导。"""

    print("\n========== 实验2：多变量求导 ==========")

    # 两个输入特征。
    x = torch.tensor([2.0, 3.0])

    # 两个可学习权重。
    w = torch.tensor(
        [1.0, 2.0],
        requires_grad=True
    )

    # 一个可学习偏置。
    b = torch.tensor(1.0, requires_grad=True)

    target = torch.tensor(12.0)

    # 线性模型：
    # y_pred = w1*x1 + w2*x2 + b
    y_pred = torch.sum(w * x) + b

    # 平方误差。
    loss = (y_pred - target) ** 2

    # 计算所有参数的梯度。
    loss.backward()

    print("预测值：", y_pred.item())
    print("损失值：", loss.item())
    print("权重梯度：", w.grad)
    print("偏置梯度：", b.grad)

    # 手动验证：
    # y_pred = 1*2 + 2*3 + 1 = 9
    # 预测误差 = 9 - 12 = -3
    #
    # dL/dw1 = 2*(-3)*2 = -12
    # dL/dw2 = 2*(-3)*3 = -18
    # dL/db  = 2*(-3)   = -6
    expected_w_grad = torch.tensor([-12.0, -18.0])
    expected_b_grad = torch.tensor(-6.0)

    assert torch.allclose(w.grad, expected_w_grad)
    assert torch.allclose(b.grad, expected_b_grad)

    print("自动求导与手算结果一致。")


def experiment_3_gradient_accumulation():
    """实验3：观察梯度累积现象。"""

    print("\n========== 实验3：梯度累积 ==========")

    w = torch.tensor(2.0, requires_grad=True)

    # 第一次构建计算图并反向传播。
    loss1 = w ** 2
    loss1.backward()

    print("第一次 backward 后：", w.grad.item())

    # 第二次重新构建计算图。
    # 注意不能直接对之前的 loss1 再次 backward，
    # 因为它的中间求导图默认已经被释放。
    loss2 = w ** 2
    loss2.backward()

    print("第二次 backward 后：", w.grad.item())

    # 当前梯度为 4+4=8。
    assert torch.allclose(w.grad, torch.tensor(8.0))

    # 将梯度清零。
    w.grad.zero_()

    print("清零后的梯度：", w.grad.item())

    # 再次求导，新的梯度为4。
    loss3 = w ** 2
    loss3.backward()

    print("清零后重新求导：", w.grad.item())

    assert torch.allclose(w.grad, torch.tensor(4.0))


def experiment_4_detach_no_grad():
    """实验4：理解 detach 和 no_grad 的区别。"""

    print("\n========== 实验4：detach 与 no_grad ==========")

    w = torch.tensor(2.0, requires_grad=True)

    # 普通可微运算。
    a = w * 3

    print("a.requires_grad：", a.requires_grad)
    print("a.grad_fn：", a.grad_fn)

    # 将 a 从原计算图分离。
    b = a.detach()

    print("b.requires_grad：", b.requires_grad)
    print("b.grad_fn：", b.grad_fn)

    # b 本身不追踪梯度。
    # 因此不能直接对一个只依赖 b 的标量执行 backward。
    #
    # 分离不会修改原来的 a。
    loss = a ** 2
    loss.backward()

    print("通过 a 反向传播，w.grad：", w.grad.item())

    # 手算：
    # a = 3w
    # L = a^2 = 9w^2
    # dL/dw = 18w = 36
    assert torch.allclose(w.grad, torch.tensor(36.0))

    # 关闭梯度记录，适合预测与参数更新等场景。
    with torch.no_grad():
        c = w * 5

    print("no_grad 中 c.requires_grad：", c.requires_grad)

    # 注意：
    # no_grad 只影响代码块内通常的梯度记录，
    # 不会永久修改 w.requires_grad。
    print("w.requires_grad：", w.requires_grad)


def experiment_5_linear_layer_gradient():
    """
    实验5：验证线性层的矩阵梯度。

    数学公式采用单样本列向量约定：
        y = Wx + b
        dL/dW = delta * x.T

    为方便理解，输入为：
        x = [1, 2, 3]

    权重形状：
        W = (2, 3)

    输出形状：
        y = (2,)
    """

    print("\n========== 实验5：矩阵梯度验证 ==========")

    # 三个输入特征。
    x = torch.tensor([1.0, 2.0, 3.0])

    # 权重矩阵：两个输出神经元、三个输入特征。
    W = torch.tensor([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0]
    ], requires_grad=True)

    # 两个输出神经元各有一个偏置。
    b = torch.tensor(
        [1.0, 2.0],
        requires_grad=True
    )

    # 目标输出。
    target = torch.tensor([10.0, 5.0])

    # 前向传播。
    # 这里 x 是一维张量，
    # W @ x 的结果也是一维张量。
    y_pred = W @ x + b

    # 为了和手算一致，使用平方误差和的一半。
    loss = 0.5 * torch.sum((y_pred - target) ** 2)

    # 自动求导。
    loss.backward()

    print("预测值：", y_pred)
    print("目标值：", target)
    print("损失：", loss.item())
    print("\n自动计算的权重梯度：")
    print(W.grad)

    # 手动计算损失对输出的梯度。
    # 对本损失函数，delta = y_pred - target。
    delta = y_pred.detach() - target

    # 使用外积计算权重梯度。
    #
    # delta.reshape(2,1) -> (2,1)
    # x.reshape(1,3)     -> (1,3)
    #
    # (2,1) @ (1,3) -> (2,3)
    manual_grad = (
        delta.reshape(2, 1)
        @ x.reshape(1, 3)
    )

    print("\n手动计算的权重梯度：")
    print(manual_grad)

    # 检查手动结果与自动求导是否一致。
    assert torch.allclose(W.grad, manual_grad)

    # 偏置梯度等于 delta。
    assert torch.allclose(b.grad, delta)

    print("\n验证通过：自动求导与手动外积计算一致。")


def main():
    """按顺序执行五个自动求导实验。"""

    experiment_1_requires_grad()
    experiment_2_multivariable()
    experiment_3_gradient_accumulation()
    experiment_4_detach_no_grad()
    experiment_5_linear_layer_gradient()

    print("\n全部 Autograd 实验执行完成！")


if __name__ == "__main__":
    main()
