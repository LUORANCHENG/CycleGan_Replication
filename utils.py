import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image

# -------------------------------------------------------存储并管理生成器产生的假样本--------------------------------------
class ReplayBuffer:
    """
    作用：
        存储并管理生成器产生的假样本
    """
    def __init__(self, max_size=50):
        # 检查 max_size 是否大于 0
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."

        self.max_size = max_size

        # 用于存储生成的假样本
        self.data = []

    def push_and_pop(self, data):
        # 用于保存最后要返回的假样本，用于训练判别器
        to_return = []

        # 遍历data中的所有假样本
        for element in data.data:
            # 将element在第 0 维扩展一个维度
            element = torch.unsqueeze(element, 0)
            # 如果缓冲区中样本数量还没有达到 max_size
            if len(self.data) < self.max_size:
                # 将生成的假样本 element 存储到缓冲区中
                self.data.append(element)
                # 同时这个假样本也要返回，用于训练判别器
                to_return.append(element)
            else:
                # 生成一个在[0, 1]之间的随机数，如果大于0.5，则从缓冲区中取出一个历史样本
                if random.uniform(0, 1) > 0.5:
                    # 从缓冲区中随机选取一个索引i
                    i = random.randint(0, self.max_size - 1)
                    # 将第i个假样本返回，用于训练判别器
                    to_return.append(self.data[i].clone())
                    # 用新生成的样本替换掉缓冲区中的第i个样本
                    self.data[i] = element
                else:
                    # 直接返回当前的假样本，用于训练判别器
                    to_return.append(element)
        return Variable(torch.cat(to_return))
# -------------------------------------------------------存储并管理生成器产生的假样本--------------------------------------


# ----------------------------------------------------自定义学习率更新策略--------------------------------------------------
class LambdaLR:
    """
    作用：
        为学习率调度器定义一个自定义的学习率更新策略
    """
    def __init__(self, n_epochs, offset, decay_start_epoch):
        """
        输入：
            n_epochs：总共的训练epoch数
            offset：训练已经进行的epoch偏移量，通常用于从中断点恢复训练时，确保衰减从正确的时刻开始
            decay_start_epoch：表示从哪一个epoch开始衰减学习率
        """

        # 保衰减开始的 epoch 小于总的训练 epoc
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):

        # 在decay_start_epoch之前，学习率不衰减，始终为1
        # 在decay_start_epoch之后，学习率开始衰减，将线性衰减因子从 1.0 开始，逐渐减小到 0.0。
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
# ----------------------------------------------------自定义学习率更新策略--------------------------------------------------
