import argparse
import random

from utils.dataloader import Dataloader
from utils.chart import Datashow

def main():
    parser = argparse.ArgumentParser(description="Datasets make....")
    parser.add_argument('--backbone', type=str, default='slr',
                        choices=['slr', 'plsr'],
                        help='backbone name (default: slr)')
    
    args = parser.parse_args()
    dataloader = Dataloader()

    # 加载所有数据
    sampledata = list(dataloader.ldsampleinfo())
    # 将所有数据随机打乱
    random.shuffle(sampledata)
    # 按比例随机划分训练集、验证集、测试集
    train_size = int(0.7 * len(sampledata))
    val_size = int(0.1 * len(sampledata))
    train_data = sampledata[:train_size]
    val_data = sampledata[train_size:train_size+val_size]
    test_data = sampledata[train_size+val_size:]

    # 打开并写入文件
    with open('./datasets/train.txt', mode = 'w') as train_file:
        for data in train_data:
            train_file.write(str(data[0]) + '\t' + str(data[1]) + '\n')
        train_file.close()
    with open('./datasets/val.txt', mode = 'w') as val_file:
        for data in val_data:
            val_file.write(str(data[0]) + '\t' + str(data[1]) + '\n')
        val_file.close()
    with open('./datasets/test.txt', mode = 'w') as test_file:
        for data in test_data:
            test_file.write(str(data[0]) + '\t' + str(data[1]) + '\n')
        test_file.close()

    print(args)

if __name__ == "__main__":
    main()
