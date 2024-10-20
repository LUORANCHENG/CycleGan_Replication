## Installation
    $ git clone https://github.com/LUORANCHENG/CycleGan.git
    $ cd CycleGan/
    $ pip install -r requirements.txt

## run
    $ cd data/
    $ bash download_cyclegan_dataset.sh monet2photo
    $ cd ../implementations/cyclegan/
    $ python3 cyclegan.py --dataset_name monet2photo
