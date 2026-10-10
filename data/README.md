# 数据

程序通过 `torchvision.datasets.MNIST` 自动将 MNIST 下载到 `data/MNIST/`。原始训练集包含 60,000 张图像，测试集包含 10,000 张图像，均为 28 × 28 像素的灰度图像。

原始数据和派生张量不纳入版本控制。首个样本按数据集原始顺序确定。标签仅用于选择样例和汇总结果，不作为模型输入或训练目标。

数据接口：[Torchvision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)。
