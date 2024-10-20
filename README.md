## 安装
    $ conda create -n CycleGan python=3.8
    $ conda activate CycleGan
    $ git clone https://github.com/LUORANCHENG/CycleGan.git
    $ cd CycleGan/
    $ pip install -r requirements.txt

## 运行
linux系统下可通过bash脚本下载数据集:

    $ cd datasets/
    $ bash download_cyclegan_dataset.sh monet2photo

windwos系统则需要手动下载数据集

复制网址`http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/`到浏览器打开选择自己喜欢的数据集下载，并调整为如下所示的文件目录结构

下载好的数据集目录应该为
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
回到项目根目录下

    $ python cyclegan.py --dataset_name monet2photo
