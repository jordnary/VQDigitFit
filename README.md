# VQDigitFit

使用 PyTorch 实现的 MNIST 向量量化自编码器（VQ-VAE）。模型将手写数字编码为 **7 × 7 的离散索引网格**，通过包含 **64 个 32 维向量**的可学习码本重建图像，重建目标为逐像素 **L1 损失**。

项目提供完整的训练、独立测试、受控扰动实验和可视化流程。训练仅使用原始训练集中的 60,000 张图像；测试集中的 10,000 张图像仅用于最终评估。数字标签不参与模型优化。

## 模型与实验

```text
Image                  Encoder features         Quantized features       Reconstruction
N × 1 × 28 × 28   →    N × 32 × 7 × 7     →    N × 32 × 7 × 7      →    N × 1 × 28 × 28
                                 ↓                    ↑
                              49 indices → Codebook: 64 × 32
```

- 编码器与解码器采用卷积、残差块和转置卷积，量化使用平方欧氏距离。
- 优化目标为 L1 重建损失、码本损失和权重为 0.25 的承诺损失之和；后两项采用均方误差。
- 直通估计器将重建梯度传回编码器，码本由独立的码本损失更新。
- 解码器使用线性输出，所有数值指标直接使用原始输出。灰度图显示范围固定为 `[0, 1]`。
- 在数据集原始顺序下选择每类数字的首个样本，并对数字 5 比较高斯噪声、第二近邻量化和随机码字量化。
- 提供训练曲线、码本利用率、独立测试指标，以及随机量化的 100 次重复统计。

## 快速开始

需要 Python 3.10 或更高版本。安装与计算设备兼容的 [PyTorch 和 Torchvision](https://pytorch.org/get-started/locally/)，再安装项目依赖：

```shell
python -m pip install -r requirements.txt
python src/vqdigitfit.py
```

命令默认下载 MNIST，使用完整训练集训练 40 轮，然后完成全部评估。数据写入 `data/`，权重、指标和图片写入 `outputs/`。程序在可用时自动选择 CUDA，也支持 CPU。

```shell
# 仅训练
python src/vqdigitfit.py --mode train

# 使用最终权重重新生成评估结果
python src/vqdigitfit.py --mode evaluate

# 指定计算设备或独立输出目录
python src/vqdigitfit.py --device cpu --output outputs/cpu_run

# 查看完整参数
python src/vqdigitfit.py --help

# 检查量化语义、梯度隔离、随机复现及指标定义
python -m unittest discover -s tests -v
```

训练会覆盖所选输出目录中的同名权重和训练历史。每轮保存模型，评估入口只接受完成预定轮数的最终权重；当前不提供中断后恢复优化器状态的功能。

## 默认配置

| 参数 | 默认值 |
| --- | --- |
| 训练轮数 | 40 |
| 批大小 | 256 |
| 优化器 | Adam，默认动量系数，无权重衰减 |
| 学习率 | 初值 `5e-4`，余弦退火下限 `5e-5` |
| 承诺损失权重 | 0.25 |
| 模型初始化与数据打乱种子 | 42 |
| 噪声种子 | 43 |
| 随机量化种子 | 44 |
| 高斯噪声标准差 | `5/255` |
| 随机量化重复次数 | 100 |

每轮遍历所有训练样本，不丢弃最后一个批次；最终轮权重用于评估，不根据测试指标选择权重。程序启用确定性算法并禁用 TF32；不同软件版本或计算后端之间仍可能存在浮点差异。

## 实验定义

**首个样本**指对应训练集或测试集按原始顺序出现的首张指定数字图像，与训练时的随机打乱顺序无关。两个分区分别保存结果，具体零基索引记录在 `metrics.json` 中。

**高斯噪声**在除以 255 后的图像上添加，满足 `epsilon ~ N(0, (5/255)^2)`。加噪输入不裁剪，以保持所定义的加性高斯扰动；图像显示会截断超出灰度范围的值。噪声输入和重建的指标均以干净图像为参照。

**第二近邻量化**在每个空间位置对全部 64 个码字按距离排序，选择第 2 个。相同距离按码字索引排序，保证结果可复现。

**随机码字量化**在每个空间位置独立、均匀地选择一个已训练码本中的码字，允许重复。固定模型参数与码本，只替换量化选择规则。该实验不重新初始化码本，也不重新训练解码器。

**评估指标**包括逐图 MAE、MSE、PSNR，以及完整测试集的汇总结果。`psnr_db` 是逐图 PSNR 的算术平均，`psnr_from_global_mse_db` 由全局 MSE 换算，两者分别记录。PSNR 使用单位峰值，MSE 下限为 `1e-12`。码本困惑度根据整个测试集的索引频率计算。

## 文件结构

```text
.
├── src/
│   └── vqdigitfit.py       # 训练、量化、评估与绘图
├── tests/
│   └── test_vqdigitfit.py  # 关键数值行为检查
├── data/                  # MNIST 下载目录
├── outputs/               # 权重、指标、图片与数值样例
├── reports/               # 独立报告及其插图，不纳入 Git
├── requirements.txt
└── README.md
```

数据、模型权重、运行产物和报告均由 `.gitignore` 排除。核心实现为单个 Python 文件，少于 500 行；不需要额外训练脚本或外部模型。

## 参考资料

- van den Oord, Vinyals, Kavukcuoglu. [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937).
- PyTorch. [L1Loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html).
- Torchvision. [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html).
