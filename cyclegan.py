import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import warnings

# 忽略一些警告
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------处理命令行参数---------------------------------------------------------------------
# 初始化一个参数解析器，用于从命令行中读取用户输入的参数
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="指定训练从第几轮开始")
parser.add_argument("--n_epochs", type=int, default=200, help="指定训练的总轮数")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="指定数据集的名称")
parser.add_argument("--batch_size", type=int, default=1, help="指定训练时的批次大小")
parser.add_argument("--lr", type=float, default=0.0002, help=" 指定Adam优化器的学习率")
parser.add_argument("--b1", type=float, default=0.5, help="指定Adam优化器的一阶动量衰减参数")
parser.add_argument("--b2", type=float, default=0.999, help="指定Adam优化器的二阶动量衰减参数")
parser.add_argument("--decay_epoch", type=int, default=100, help="指定从第几轮开始进行学习率衰减")
parser.add_argument("--n_cpu", type=int, default=8, help="指定用于数据加载时的 CPU 线程数量")
parser.add_argument("--img_height", type=int, default=256, help="指定输入图像的高度")
parser.add_argument("--img_width", type=int, default=256, help="指定输入图像的宽度")
parser.add_argument("--channels", type=int, default=3, help="指定图像的通道数")
parser.add_argument("--sample_interval", type=int, default=100, help="指定每隔多少步保存一次生成器的输出图像")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="指定保存模型检查点的间隔")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="指定生成器中残差块的数量")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="指定循环一致性损失的权重")
parser.add_argument("--lambda_id", type=float, default=5.0, help="指定身份损失的权重")

# 将用户输入的命令行参数解析为 opt 对象，并可以在程序中使用这些参数
opt = parser.parse_args()
print(opt)
# -----------------------------------------------------------------------处理命令行参数---------------------------------------------------------------------

# --------------------------------------------------------------------------初始化------------------------------------------------------------------------
# 创建用于保存生成图像的文件夹
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)

# 创建用于保存训练过程中生成的模型权重的文件夹
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# 使用均方误差来计算判别器输出的结果和目标标签的差异
criterion_GAN = torch.nn.MSELoss()

# 使用平均绝对误差来评估循环一致性损失
criterion_cycle = torch.nn.L1Loss()

# 使用平均绝对误差来计算身份损失
criterion_identity = torch.nn.L1Loss()

# 检查cuda是否可用
cuda = torch.cuda.is_available()

# 图像的输入尺寸
input_shape = (opt.channels, opt.img_height, opt.img_width)

# 创建生成器 G_AB，它的作用是将图像从域A（例如：马的图片）转换为域B（例如：斑马的图片）
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)

# 创建生成器 G_BA，它的作用是将图像从域B（斑马的图片）转换回域A（马的图片）
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)

# 创建了判别器 D_A，它的作用是判断输入的图像是否属于域A（例如：马的图片）或是由生成器生成的“假”马的图像
D_A = Discriminator(input_shape)

# 创建了另一个判别器 D_B，它的作用是判断输入的图像是否属于域B（例如：斑马的图片）或是由生成器生成的“假”斑马的图像
D_B = Discriminator(input_shape)

# 如果cuda可用，则把所有的模型和优化器转移到cuda上
if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

# 如果 opt.epoch 不等于 0，表示用户希望从一个特定的训练轮次开始继续训练，而不是从头开始
if opt.epoch != 0:
    # 加载之前保存的预训练模型
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # 初始化生成器和判别器的权重
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# 定义生成器的优化器，将生成器 G_AB 和 G_BA 的所有参数（权重）链在一起，作为一个整体进行优化
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

# 定义判别器 D_A 的优化器
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 定义判别器 D_B 的优化器
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 初始化学习率调度器，根据训练的进展动态调整学习率
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

# 如果有GPU，模型就使用GPU加速的张量类型；如果没有GPU，模型就使用CPU版本的张量
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


# 创建两个重放缓存，分别用于存储生成的“假A”和“假B”图像
"""
在GAN的训练中，如果直接使用生成器最新生成的图像去训练判别器，判别器可能会快速适应生成器
的输出，导致训练不稳定。引入重放缓存的目的，是通过存储以前生成的“假”图像，在训练判别器时
不仅使用当前生成的图像，还随机使用一些之前生成的图像，这样可以打破生成器和判别器之间的快
速适应，提升模型的训练效果。
"""
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
# --------------------------------------------------------------------------初始化------------------------------------------------------------------------


# ---------------------------------------------------------------------对图像进行预处理操作------------------------------------------------------------------
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),  # 调整图像的大小，将图像的高度和宽度按比例缩放
    transforms.RandomCrop((opt.img_height, opt.img_width)),  # 随机裁剪图像到指定的大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像，即左右对称翻转
    transforms.ToTensor(),  # 转换为PyTorch的张量，并将像素值从0-255的范围归一化到0-1之间
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 将图像的像素值归一化到[-1, 1]的范围，便于模型训练
]
# ---------------------------------------------------------------------对图像进行预处理操作------------------------------------------------------------------



# ---------------------------------------------------------------------初始化数据集加载器--------------------------------------------------------------------
# 训练集数据加载器
dataloader = DataLoader(
    ImageDataset("./datasets/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# 测试集数据加载器
val_dataloader = DataLoader(
    ImageDataset("./datasets/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)
# ---------------------------------------------------------------------初始化数据集加载器--------------------------------------------------------------------

def sample_images(batches_done):
    """
    作用：
        从验证数据集中获取一批图像，生成其对应的转换结果，并将这些图像拼成网格保存起来
    输入：
        batches_done：这是一个整数，表示当前训练过程中已经完成的批次数。它用于命名保存的图像文件
    输出：
        无返回值，但它会将生成的图像网格保存到磁盘中
    """

    # 从验证数据集中获取一批图像。imgs 是一个包含真实图像A和B的批次数据
    imgs = next(iter(val_dataloader))

    # 将两个生成器 G_AB 和 G_BA 切换到评估模式
    G_AB.eval()
    G_BA.eval()

    # 将真实的域A图像real_A转换为PyTorch张量
    real_A = Variable(imgs["A"].type(Tensor))

    # 使用生成器 G_AB 生成“假B”图像 fake_B，即从域A（如马）转换到域B（如斑马）的图像
    fake_B = G_AB(real_A)

    # 将真实的域B图像real_B转换为PyTorch张量
    real_B = Variable(imgs["B"].type(Tensor))

    # 使用生成器 G_BA 生成“假A”图像 fake_A，即从域B（如斑马）转换到域A（如马）的图像
    fake_A = G_BA(real_B)

    # 将图像排列成网格
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)

    # 使用 torch.cat 沿着y轴（第二个维度）将 real_A、fake_B、real_B 和 fake_A 这四个图像网格连接起来
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

    # 将生成的图像网格保存到磁盘
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)



# ----------------------------------------------------------------开始训练---------------------------------------------------
# 获取当前的时间，用于后续计算训练所耗费的时间
prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # 获取来自域A和域B的图像数据，用作模型的输入
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # 创建真实的标签，形状为(batch_size, 1, height, width)的张量，值全为1
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        # 创建虚假的标签，形状为(batch_size, 1, height, width)的张量，值全为0
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        # 将生成器设置为训练模式
        G_AB.train()
        G_BA.train()

        # --------------------------------------------生成器训练-------------------------------------
        # 梯度清零
        optimizer_G.zero_grad()

        # 计算身份损失，身份损失的作用是确保当输入图像已经属于目标域时，生成器不会对其进行不必要的修改
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        # 将两个身份损失秋平均作为最终的身份损失，这样可以平衡两个生成器的身份损失
        loss_identity = (loss_id_A + loss_id_B) / 2



        # 通过real_A图像生成fake_B图像
        fake_B = G_AB(real_A)
        # 计算生成器G_AB的GAN损失，如果loss_GAN_AB较小，说明D_B认为fake_B是真实的，生成器成功欺骗了判别器
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        # 通过real_B图像生成fake_A图像
        fake_A = G_BA(real_B)
        # 计算生成器G_BA的GAN损失，如果loss_GAN_BA较小，说明D_A认为fake_A是真实的，生成器成功欺骗了判别器
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        # 将两个损失求平均
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2


        # recov_A代表的是经过两次转换后的图像，即：从A（马）到B（斑马），再从B返回A
        recov_A = G_BA(fake_B)
        # 计算循环一致性损失，它表示从A到B再回到A的过程中，图像的恢复效果如何。
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        # recov_B代表的是经过两次转换后的图像，即：从B（斑马）到A（马），再从A返回B
        recov_B = G_AB(fake_A)
        # 计算循环一致性损失，它表示从B到A再回到B的过程中，图像的恢复效果如何。
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        # 对两个损失去平均
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # 计算最终损失
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

        # 反向传播
        loss_G.backward()

        # 更新参数
        optimizer_G.step()
        # --------------------------------------------生成器训练-------------------------------------


        # --------------------------------------------判别器A训练------------------------------------
        # 梯度清零
        optimizer_D_A.zero_grad()

        # 这是判别器在判断真实图像时的损失。如果D_A(real_A)的输出接近valid（即1，损失较小
        loss_real = criterion_GAN(D_A(real_A), valid)

        # 将新生成的fake_A图像加入缓存，同时从缓存中取出一批假图像
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)

        # 这是判别器在判断假图像时的损失。如果 D_A(fake_A_) 输出接近0，损失较小
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)

        # 将两个损失取平均作为总损失
        loss_D_A = (loss_real + loss_fake) / 2

        # 反向传播
        loss_D_A.backward()

        # 更新参数
        optimizer_D_A.step()
        # --------------------------------------------判别器A训练------------------------------------

        # --------------------------------------------判别器B训练------------------------------------
        # 梯度清零
        optimizer_D_B.zero_grad()

        # 这是判别器在判断真实图像时的损失。如果D_B(real_B)的输出接近valid（即1，损失较小
        loss_real = criterion_GAN(D_B(real_B), valid)

        # 将新生成的fake_B图像加入缓存，同时从缓存中取出一批假图像
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)

        # 这是判别器在判断假图像时的损失。如果 D_B(fake_B_) 输出接近0，损失较小
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)

        # 将两个损失取平均作为总损失
        loss_D_B = (loss_real + loss_fake) / 2

        # 反向传播
        loss_D_B.backward()

        # 参数更新
        optimizer_D_B.step()
        # --------------------------------------------判别器B训练------------------------------------

        # 判别器的总损失
        loss_D = (loss_D_A + loss_D_B) / 2


        # ------------------------------------------日志处理-------------------------------------
        # 表示已经完成的总批次数
        batches_done = epoch * len(dataloader) + i

        # 表示剩余的批次数量
        batches_left = opt.n_epochs * len(dataloader) - batches_done

        # 训练剩余时间的估计值
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

        # 更新 prev_time 为当前的时间戳
        prev_time = time.time()

        # 打印日志
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )

        # 定期保存生成的样本图像
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # 在完成一个epoch后更新学习率
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # 定期保存模型的检查点
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
        torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
    # ------------------------------------------日志处理-------------------------------------
