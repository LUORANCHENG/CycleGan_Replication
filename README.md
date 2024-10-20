## 安装
    $ conda create -n CycleGan python=3.8
    $ conda activate CycleGan
    $ git clone https://github.com/LUORANCHENG/CycleGan.git
    $ cd CycleGan/
    $ pip install -r requirements.txt

## 运行
    $ cd data/
    $ bash download_cyclegan_dataset.sh monet2photo
    $ cd ../implementations/cyclegan/
    $ python3 cyclegan.py --dataset_name monet2photo
