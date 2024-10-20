import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    """
    作用：
        将输入图像转换为 RGB 格式
    输入：
        image：输入图像对象
    输出：
        rgb_image：这是一个新的 RGB 格式 的图像对象，大小与输入图像相同
    """

    # 创建一个新的 RGB 图像，大小与输入图像 image 相同
    rgb_image = Image.new("RGB", image.size)

    # 将输入图像image粘贴到刚刚创建的rgb_image图像上。如果输入图像不是RGB模式，paste方法会自动将其转换为RGB格式
    rgb_image.paste(image)

    # 最终返回转换后的 RGB 图像对象
    return rgb_image


class ImageDataset(Dataset):
    """
    作用：
        加载和处理成对的图像数据集
    """
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        """
        输入：
            root：数据集的根目录
            transforms_：一个列表，包含要应用于图像的转换操作
            unaligned：是否加载不对齐的图像对
            mode：可以是 "train" 或 "test"
        """

        # 将传入的 transforms_ 列表组合成一个可以对图像进行转换的函数链
        self.transform = transforms.Compose(transforms_)

        self.unaligned = unaligned

        # 通过glob.glob获取目录root/train/A或root/test/A下的所有图像文件路径，并对文件名进行排序
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))

        # 同上
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        # 通过index取出A域的图像路径并打开图像
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        # 如果unaligned为True，则从B域中随机选取一张图像；如果为False，则与A域的图像按顺序对齐读取
        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # 转换灰度图像为 RGB
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

         # 执行一系列预处理或数据增强操作
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        # 返回一个字典，包含转换后的图像 item_A 和 item_B
        return {"A": item_A, "B": item_B}

    # 返回数据集的长度
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
