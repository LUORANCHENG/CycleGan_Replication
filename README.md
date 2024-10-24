## 目录：

- [参考项目](#section1)
- [项目简介](#section2)
- [安装](#section3)
- [运行](#section4)
- [原理介绍](#section5)
    - [1.什么是生成对抗网络gan](#subsection1)
    - [2.什么是CycleGan](#subsection2)


## 参考项目<a name="section1"></a>
https://github.com/junyanz/CycleGAN

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

https://github.com/eriklindernoren/PyTorch-GAN

[GAN生成对抗网络原理及cyclegan pytorch实战：天气季节转换 物种变换 医疗转换 建筑外立面转设计图_计算机视觉](https://www.bilibili.com/video/BV1qVmwYfEN4/?spm_id_from=333.337.search-card.all.click&vd_source=1a02178b1644ddc9b579739c3c1616b4)

[精读CycleGAN论文-拍案叫绝的非配对图像风格迁移](https://www.bilibili.com/video/BV1Ya411a78P/?spm_id_from=333.337.search-card.all.click&vd_source=1a02178b1644ddc9b579739c3c1616b4)


## 项目简介<a name="section2"></a>

本项目为深度学习的课程作业，使用CycleGan模型实现将现实的照片转换为莫奈风格的画
![picture2monet](https://github.com/user-attachments/assets/e44d811d-dd70-46e1-98b2-a69193a6d3c7)




## 安装<a name="section3"></a>
    $ conda create -n CycleGan python=3.8
    $ conda activate CycleGan
    $ git clone https://github.com/LUORANCHENG/CycleGan.git
    $ cd CycleGan/
    $ pip install -r requirements.txt

## 运行<a name="section4"></a>
Linux系统下可通过bash脚本下载数据集:

    $ cd datasets/
    $ bash download_cyclegan_dataset.sh monet2photo

Windows系统则需要手动下载数据集:

复制网址`http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/`到浏览器打开选择monet2photo数据集下载，并调整为如下所示的文件目录结构

下载好的数据集目录应该为:
```
datasets
 └── monet2photo
   └── train
    ├── A
    └── B
   └── test
    ├── A
    └── B
```
回到项目根目录下，执行`cyclegan.py`开始训练

    $ python cyclegan.py --dataset_name monet2photo
## 原理介绍<a name="section5"></a>
### 1.什么是生成对抗网络gan<a name="subsection1"></a>

GAN的核心思想是通过生成器和判别器之间的对抗过程，使得生成器能够生成以假乱真的数据，而判别器则努力区分真实数据和生成的数据。这个过程类似于一个零和博弈，其中两个网络不断优化自己以提高对抗能力。

GAN的基本原理:生成器和判别器

生成器G:负责从随机噪声中生成新的数据样本。它的目标是生成能够欺骗判别器的数据。

判别器D:负责判断输入的数据是真实的还是由生成器生成的。它的目标是正确地区分真实数据和生成的数据。
![image](https://github.com/user-attachments/assets/6075edc0-e85f-4a54-894d-48ea2fe1b127)


GAN的核心公式：
![gan_公式](https://github.com/user-attachments/assets/1a6039d2-90bf-4b6a-99db-bc53dd1c1de6)

从判别器角度来理解公式：
![判别器角度公式](https://github.com/user-attachments/assets/394f3b2b-352b-4985-89b6-6b1acd95fbff)

从生成器角度来理解公式：
![生成器角度公式](https://github.com/user-attachments/assets/f4f5d1e5-5091-484b-828f-07b03ca2ed99)

训练过程：

在训练开始时，生成器和判别器的能力都很弱。通过交替训练这两个网络，判别器逐渐学会区分真实数据和生成数据，而生成器则学会生成更真实的数据以欺骗判别器。

这个过程通过最小化生成器和判别器之间的差异来进行，最终目标是使判别器无法区分真实数据和生成的数据，即达到 纳什均衡

### 2.什么是CycleGan<a name="subsection2"></a>

CycleGan是在基础gan模型上实现的，是一种用于图像到图像转换的深度学习模型，它能够在没有成对训练数据的情况下，实现不同图像域之间的转换。比如说，CycleGan可以实现将一张真实的照片转换为莫奈风格的油画，同时也可以实现将莫奈风格的油画转换回真实的照片
![image](https://github.com/user-attachments/assets/dc83474c-0e9c-47fb-92c9-6b5b3a5b1832)

#### 2.1.CycleGan的基本原理：
![image](https://github.com/user-attachments/assets/0abf5024-7c6d-47c6-91ca-e0a110d5b9f6)

(a)部分：X假设为莫奈风格油画的图像数据集，Y假设为真实风景照片的图像数据集。我们需要同时训练两个生成对抗网络

- 第一个生成对抗网络：假设生成器为G，判别器为D_Y，生成器G把X域的图像x转换成Y域的图像y'，而D_Y判别器就来判别图像是来自真正的y还是由G生成出来的y'。最后训练出来的结果是生成器G能够生成以假乱真的y'图像使判别器D_Y无法区分。

- 第二个生成对抗网络：假设生成器为F，判别器为D_X，第二个生成对抗网络所干的事恰恰相反，F是将输入的Y域图像转换成X域的图像，使X域的图像足够逼真，无法让D_X来辨别真假

(b)部分：展示了循环一致性损失的作用

- 通过生成器G将X域的图像x转换为Y域的图像y'，然后我们再将y'通过生成器F转换回X域的图像x'，然后我们需要对比x和x'的误差(循环一致性损失)，使误差最小化，从而保证转换后的图像具有一致性

(c)部分：同(b)部分

#### 2.2.CycleGan的关键损失

(1)Dy处的GAN损失：

![图片](https://github.com/user-attachments/assets/9c6d026d-cb41-43b2-9abe-ca5c340ddece)

(2)Dx处的GAN损失:

![图片](https://github.com/user-attachments/assets/ae263572-1acc-4b8c-9e03-6692c3680533)

(3)循环一致性损失

![图片](https://github.com/user-attachments/assets/badaf623-7b02-4f06-9999-1771fba5c72e)

(4)Identity loss

![图片](https://github.com/user-attachments/assets/eccde6b8-15a6-4455-ac2f-d50c3b3082cc)

注：Identity loss的作用主要是为了保持图像的整体色调不发生大的变化，如果不加Identity loss，那么生成器可能会自主地修改图像的色调，使得整体的颜色产生变化，如下图所示

![图片](https://github.com/user-attachments/assets/19efc92e-36ac-4654-8761-8000e9080f4b)


#### 2.3.生成器和判别器的模型结构

<div style="text-align: center"><strong>生成器模型结构</strong></div>

![CycleGan (2)](https://github.com/user-attachments/assets/4254ce16-7bb2-4c40-a395-0b4ebc1918ba)

<div style="text-align: center"><strong>判别器模型结构</strong></div>

![CycleGan_gen (1)](https://github.com/user-attachments/assets/c0591e97-a979-4889-a59b-8dcc7fdcda4c)



