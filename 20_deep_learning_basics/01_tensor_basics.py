import torch

def section_1_create_tensor():
    """
    实验1：认识张量的维度、形状和元素总数。

    需要理解：
    1. ndim：张量有几个维度。
    2. shape：每个维度的大小。
    3. numel()：张量中共有多少个元素。
    """

    print("\n========== 实验1：创建 Tensor ==========")

    # 0维张量：标量。
    # shape 为 torch.Size([])，即空元组形状。
    scalar = torch.tensor(5.0)

    # 1维张量：长度为3的向量。
    vector = torch.tensor([1.0, 2.0, 3.0])

    # 2维张量：2行3列的矩阵。
    matrix = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    # 3维张量：包含2个矩阵，每个矩阵为2行3列。
    tensor_3d = torch.tensor([
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]],

        [[7.0, 8.0, 9.0],
         [10.0, 11.0, 12.0]]
    ])

    tensors = {
        "标量": scalar,
        "向量": vector,
        "矩阵": matrix,
        "三维张量": tensor_3d
    }

    for name, tensor in tensors.items():
        print(f"\n{name}：")
        print(tensor)

        # ndim 表示维度数量。
        print("维度数量：", tensor.ndim)

        # shape 表示每个维度的大小。
        print("形状：", tuple(tensor.shape))

        # numel() 计算元素总数。
        print("元素数量：", tensor.numel())

        # dtype 表示元素的数据类型。
        print("数据类型：", tensor.dtype)


def section_2_index_and_reshape():
    """
    实验2：张量索引和形状变换。

    重要说明：
    reshape 只改变元素的组织形状，
    不会改变元素总数。

    例如：
    原形状 (2, 3)，元素总数为6。
    新形状 (3, 2)，元素总数仍为6。
    """

    print("\n========== 实验2：索引与形状变换 ==========")

    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    print("原始张量：")
    print(x)
    print("原始形状：", tuple(x.shape))

    # Python索引从0开始。
    # x[0]：第1行。
    print("第1行：", x[0])

    # x[:, 1]：所有行的第2列。
    print("第2列：", x[:, 1])

    # x[1, 2]：第2行、第3列。
    print("指定元素：", x[1, 2].item())

    # reshape：将2行3列变成3行2列。
    reshaped = x.reshape(3, 2)

    print("\nreshape(3, 2) 后：")
    print(reshaped)
    print("新形状：", tuple(reshaped.shape))

    # unsqueeze：在指定位置添加一个长度为1的维度。
    # (2, 3) -> (1, 2, 3)
    batch = x.unsqueeze(0)

    print("\n添加批次维度后：")
    print("形状：", tuple(batch.shape))
    print(batch)

    # squeeze：删除长度为1的维度。
    restored = batch.squeeze(0)

    print("删除批次维度后：", tuple(restored.shape))


def section_3_matrix_multiplication():
    """
    实验3：矩阵乘法。

    矩阵乘法的规则：
    A的列数必须等于B的行数。

    如果：
    A.shape = (2, 3)
    B.shape = (3, 2)

    那么：
    C = A @ B
    C.shape = (2, 2)

    注意：
    @ 是矩阵乘法。
    * 是逐元素乘法。
    两者并不相同。
    """

    print("\n========== 实验3：矩阵乘法 ==========")

    A = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    B = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    C = A @ B

    print("A的形状：", tuple(A.shape))
    print("B的形状：", tuple(B.shape))

    print("\n矩阵乘法结果：")
    print(C)
    print("结果形状：", tuple(C.shape))

    # 验证结果矩阵左上角元素：
    # C[0,0] = 1*1 + 2*3 + 3*5 = 22
    print("\n左上角元素：", C[0, 0].item())

    # 转置会交换矩阵的行和列。
    # A: (2, 3) -> A.T: (3, 2)
    print("A转置后的形状：", tuple(A.T.shape))


def section_4_broadcasting():
    """
    实验4：广播机制（Broadcasting）。

    场景：
    一个矩阵有3行2列，
    一个偏置向量只有2个元素。

    PyTorch可以把偏置向量分别加到矩阵每一行。

    广播的一般规则：
    从最后一个维度开始比较。
    两个维度相等，或者其中一个等于1时，
    才可以在该维度进行广播。

    缺失的前导维度按1处理。
    """

    print("\n========== 实验4：广播机制 ==========")

    # 假设有3个样本，每个样本包含2个输出。
    outputs = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    # 每个输出神经元对应一个偏置。
    bias = torch.tensor([10.0, 20.0])

    # outputs.shape = (3, 2)
    # bias.shape = (2,)
    #
    # 广播时 bias 按 (1, 2) 看待，
    # 在第0维扩展到3行。
    result = outputs + bias

    print("outputs形状：", tuple(outputs.shape))
    print("bias形状：", tuple(bias.shape))
    print("\n广播相加结果：")
    print(result)

    # 使用手动复制的方式验证广播结果。
    # repeat(3, 1)表示沿第0维复制3次，
    # 沿第1维复制1次。
    expanded_bias = bias.unsqueeze(0).repeat(3, 1)

    manual_result = outputs + expanded_bias

    # torch.allclose 判断两组浮点数是否近似相等。
    print("\n广播与手动复制结果一致：",
          torch.allclose(result, manual_result))


def section_5_neural_network_batch():
    """
    实验5：模拟神经网络中的批量线性计算。

    假设：
    batch_size = 4：一次输入4个样本。
    in_features = 3：每个样本有3个特征。
    out_features = 2：每个样本输出2个数值。

    PyTorch nn.Linear 的权重形状是：
    (out_features, in_features)

    采用行向量批量表示时：
    Y = X @ W.T + b

    形状变化：
    (4, 3) @ (3, 2) + (2,)
             |
             v
           (4, 2)
    """

    print("\n========== 实验5：批量神经网络计算 ==========")

    # 4个样本，每个样本有3个输入特征。
    X = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0]
    ])

    # 输出层有2个神经元。
    # 每个神经元需要3个输入权重。
    W = torch.tensor([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0]
    ])

    # 两个输出神经元各有一个偏置。
    b = torch.tensor([1.0, 2.0])

    # 注意：这里必须使用 W.T。
    # X.shape = (4, 3)
    # W.T.shape = (3, 2)
    Y = X @ W.T + b

    print("输入X：")
    print(X)
    print("\n权重W：")
    print(W)
    print("\n偏置b：")
    print(b)
    print("\n最终输出Y：")
    print(Y)

    print("\n输入形状：", tuple(X.shape))
    print("权重形状：", tuple(W.shape))
    print("转置权重形状：", tuple(W.T.shape))
    print("输出形状：", tuple(Y.shape))

    # 使用 PyTorch 自带的线性层验证。
    layer = torch.nn.Linear(
        in_features=3,
        out_features=2
    )

    # 覆盖默认随机初始化的参数，
    # 保证它们和手动计算使用的参数一致。
    with torch.no_grad():
        layer.weight.copy_(W)
        layer.bias.copy_(b)

    Y_layer = layer(X)

    print("\nnn.Linear 的输出：")
    print(Y_layer)

    print("\n手动计算与nn.Linear一致：",
          torch.allclose(Y, Y_layer))


def main():
    """按照顺序执行全部实验。"""

    section_1_create_tensor()
    section_2_index_and_reshape()
    section_3_matrix_multiplication()
    section_4_broadcasting()
    section_5_neural_network_batch()


if __name__ == "__main__":
    main()
