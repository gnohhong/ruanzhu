import numpy as np
import argparse
import torch
from utils.dataloader import Dataloader
from utils.chart import Datashow
from utils.saver import ModelSaver
from algorism.pretreatment import *
from algorism.evaluation import *
from netmodel.mlre import *
from netmodel.mlnet import *
from netmodel.elman import *
from netmodel.tnet import *
from netmodel.ann import *

def main():
    parser = argparse.ArgumentParser(description="PyTorch Net Training....")
    parser.add_argument('--backbone', type=str, default='slr',
                        choices=['slr', 'plsr', 'lssvr', 'elm', 'ann', 'elman', 'tnet'],
                        help='backbone name (default: slr)')
    parser.add_argument('--pcasize', type=int, default=15,
                        help='choose size of pca, fit to (backbone) plsr')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 50)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.000025, metavar='LR',
                        help='learning rate (default: auto)')
    # cuda, seed and logging
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')

    args = parser.parse_args()
    print(args)

    saver = ModelSaver()
    dataloader = Dataloader()
    _size, _wave, _dark = dataloader.get_baseinfo() # 采样点，波长，暗底
    _stand = dataloader.get_standinfo() # 标准光谱
    datashow = Datashow(_wave, _dark, _stand)
    pretreat = Pretreatment(sg=True, snv=False, msc=True, dt=False, maxmin=False, mean=False, 
                                vector=True, d1=False, d2=False, wave=False)
    evaluate = Evaluation(ppmcc = True, mse = False, rmse = True, mae = True, 
                            rd = True, r2 = True, rpd = True)
    
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        
    train = dataloader.loadtrain() # 训练集
    val = dataloader.loadval() # 验证集

    if args.backbone == 'slr':
        data = dataloader.ldintensallinfo(dark = True, stand = True, save=False)
        data = pretreat.execute(data) # 光谱预处理
        slr = SLR()
        pred, real = slr.format(data, dataloader.get_wave1430_id(), 1)
        #datashow.showdata(data)
    elif args.backbone == 'plsr': # 偏最小二乘法
        plsr = PLSR(args.pcasize, 'Group')
        train = dataloader.loadtrain()
        trainval = train + dataloader.loadval() # 训练集 + 验证集
        for sample in trainval:
            data = dataloader.ldintensinfo(sample, dark = True, stand = True)
            data = pretreat.execute(data) # 光谱预处理
            plsr.loaddata(data)
        pred, real = plsr.fitpara()
        saver.save(args, plsr.getmodel())
    elif args.backbone == 'lssvr': # 最小二乘支持向量回归
        lssvr = LSSVR('Group')
        train = dataloader.loadtrain()
        trainval = train + dataloader.loadval() # 训练集 + 验证集
        for sample in trainval:
            data = dataloader.ldintensinfo(sample, dark = True, stand = True)
            data = pretreat.execute(data) # 光谱预处理
            lssvr.loaddata(data)
        pred, real = lssvr.fitpara()
        saver.save(args, lssvr.getmodel())
    elif args.backbone == 'ann':
        ann = ANNModel(args)
        for sample in train + val:
            data = dataloader.ldintensinfo(sample, dark = True, stand = True)
            data = pretreat.execute(data) # 光谱预处理
            ann.loaddata(data)
        pred, real = ann.train()
        ann.showanalysis()
        saver.save(args, ann.getmodel())
    elif args.backbone == 'elman': # ElmanRNN
        elman = ElmanModel(args)
        for sample in train + val:
            data = dataloader.ldintensinfo(sample, dark = True, stand = True)
            data = pretreat.execute(data) # 光谱预处理
            elman.loaddata(data)
        pred, real = elman.train()
        elman.showanalysis()
        saver.save(args, elman.getmodel())
    elif args.backbone == 'tnet':
        tnet = TNetModel(args)
        for sample in train + val:
            data = dataloader.ldintensinfo(sample, dark = True, stand = True)
            data = pretreat.execute(data) # 光谱预处理
            tnet.loaddata(data)
        pred, real = tnet.train()
        tnet.showanalysis()
        saver.save(args, tnet.getmodel())
    
    # 评价指标
    evaluate.execute(pred, real)
    datashow.show_rescompare(pred, real)

if __name__ == "__main__":
    main()