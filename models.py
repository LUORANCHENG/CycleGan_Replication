import torch.nn as nn
import torch.nn.functional as F
import torch

# ---------------------------------------------------------权重初始化函数-------------------------------------------------
def weights_init_normal(m):
    """
    作用：
        用于初始化神经网络权重
    输入：
        m：神经网络中的某一层模块。这个函数通常会被应用到整个神经网络的每一层上，通过遍历网络的所有层进行参数初始化
    输出：
        无返回值。但该函数会直接修改输入层 m 的权重和偏置，即完成权重的初始化操作
    """

    # 提取层的类名字符串
    classname = m.__class__.__name__

    # 判断该层是否是卷积层
    if classname.find("Conv") != -1:

        # 对卷积层的权重进行正态分布初始化
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

        # 检查该层是否有偏置项
        if hasattr(m, "bias") and m.bias is not None:

            # 将偏置项初始化为常数 0.0
            torch.nn.init.constant_(m.bias.data, 0.0)

    # 判断该层是否是归一化层
    elif classname.find("BatchNorm2d") != -1:

        # 对归一化层的权重进行正态分布初始化
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)

        # 将偏置项初始化为常数 0.0
        torch.nn.init.constant_(m.bias.data, 0.0)
# ---------------------------------------------------------权重初始化函数-------------------------------------------------



# ----------------------------------------------------resnet中的残差块-----------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),  # 反射填充，用于在输入的边缘填充1个像素，用于保持输出的大小不变
            nn.Conv2d(in_features, in_features, 3),  # 二维卷积层
            nn.InstanceNorm2d(in_features),  # 归一化层，对每个样本（单个图像）的通道分别进行归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.ReflectionPad2d(1),  # 反射填充
            nn.Conv2d(in_features, in_features, 3),  # # 二维卷积层
            nn.InstanceNorm2d(in_features),  # # 归一化层，对每个样本（单个图像）的通道分别进行归一化
        )

    def forward(self, x):
        return x + self.block(x)
# ----------------------------------------------------resnet中的残差块-----------------------------------------------------

# ------------------------------------------------------生成器模型---------------------------------------------------------
class GeneratorResNet(nn.Module):
    """
    作用：
        将输入图像转换为具有相同空间维度的输出图像
    输入：
        输入内容：图像（通常是某种风格的图像）。
        输入形状：(batch_size, channels, height, width)。
    输出：
        输出内容：生成的图像（与输入图像同样尺寸的目标风格图像）。
        输出形状：与输入形状相同，即 (batch_size, channels, height, width)。
    """
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        # 获取图像的通道
        channels = input_shape[0]

        # 初始化卷积块
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),  # 反射填充
            nn.Conv2d(channels, out_features, 7),  # 二维卷积层
            nn.InstanceNorm2d(out_features),  # 实例归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
        ]
        in_features = out_features

        # 下采样，这里进行了两次下采样操作，使图像的空间尺寸（height 和 width）逐渐减小
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),  # 二维卷积层
                nn.InstanceNorm2d(out_features),  # 实例归一化
                nn.ReLU(inplace=True),  # ReLU激活函数
            ]
            in_features = out_features

        # 残差块，这里使用了num_residual_blocks个残差块
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # 上采样，这里进行了两次上采样操作，使图像的空间尺寸恢复到原始大小
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
# ------------------------------------------------------生成器模型---------------------------------------------------------



# ------------------------------------------------------判别器模型---------------------------------------------------------

class Discriminator(nn.Module):
    """
    作用：
        用于判断输入图像是真实的还是生成的（假图像）。采用的是 PatchGAN 判别器，它不是
        对整个图像进行二分类，而是对图像的局部区域（patch）进行真假判别。
    输入：
        输入内容：图像（可能是真实图像或生成器生成的图像）。
        输入形状：(batch_size, channels, height, width)。
    输出：
        输出内容：图像的局部区域是否为真实图像的概率。
        输出形状：(batch_size, 1, height // 16, width // 16)
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        # 获取图像的通道，宽，高
        channels, height, width = input_shape

        # 这个属性保存判别器的输出形状，输出是输入图像经过四次下采样（步长为2）后的空间尺寸
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        # 判别器块的定义
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 判别器的模型结构
        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),  # 在图像的边缘进行不对称的零填充
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
# ------------------------------------------------------判别器模型---------------------------------------------------------
