## 安装
    $ conda create -n CycleGan python=3.8
    $ conda activate CycleGan
    $ git clone https://github.com/LUORANCHENG/CycleGan.git
    $ cd CycleGan/
    $ pip install -r requirements.txt

## 运行
    $ cd datasets/
    $ bash download_cyclegan_dataset.sh monet2photo
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
