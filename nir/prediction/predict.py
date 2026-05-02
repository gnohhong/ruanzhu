import numpy as np
import argparse
import os

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
import torch.onnx

def main():
    parser = argparse.ArgumentParser(description="PyTorch Net Predict....")
    parser.add_argument('--backbone', type=str, default='slr',
                        choices=['slr', 'plsr', 'lssvr', 'elm', 'ann', 'elman', 'tnet'],
                        help='backbone name (default: slr)')
    parser.add_argument('--pcasize', type=int, default=10,
                        help='choose size of pca, fit to (backbone) plsr')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='This parameter are not used')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.004, metavar='LR',
                        help='This parameter are not used')
    # cuda, seed and logging
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')

    args = parser.parse_args()
    saver = ModelSaver()
    dataloader = Dataloader()
    _size, _wave, _dark = dataloader.get_baseinfo() # 采样点，波长，暗底
    _stand = dataloader.get_standinfo() # 标准光谱
    datashow = Datashow(_wave, _dark, _stand)
    pretreat = Pretreatment(sg=True, snv=False, msc=True, dt=False, maxmin=False, mean=False, 
                                vector=True, d1=False, d2=False, wave=True)
    evaluate = Evaluation(ppmcc = True, mse = False, rmse = True, mae = True, 
                            rd = True, r2 = True, rpd = True)

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        
    test = dataloader.loadtest()
    #test = dataloader.loadtrain()
    #test = test + dataloader.loadval() # 训练集 + 验证集

    if args.backbone == 'slr':
        data = dataloader.ldintensallinfo(dark = True, stand = True, save=False)
        data = pretreat.execute(data) # 光谱预处理
        slr = SLR()
        pred, real = slr.format(data, dataloader.get_wave1430_id(), 7)
    elif args.backbone == 'plsr': # 偏最小二乘法
        plsr = PLSR(args.pcasize, 'Group')
        for sample in test:
            data = dataloader.ldintensinfo(sample, dark = True, stand = True)
            data = pretreat.execute(data) # 光谱预处理
            plsr.loaddata(data)
        plsr.setmodel(saver.load(args, plsr.getmodel()))
        pred, real = plsr.format()
    elif args.backbone == 'lssvr': # 最小二乘支持向量回归
        lssvr = LSSVR('Group')
        for sample in test:
            data = dataloader.ldintensinfo(sample, dark = True, stand = True)
            data = pretreat.execute(data) # 光谱预处理
            lssvr.loaddata(data)
        lssvr.setmodel(saver.load(args, lssvr.getmodel()))
        pred, real = lssvr.format()
    elif args.backbone == 'elm': # 极限学习机
        elm = ELM(100, 'relu', 'Group')
        # train
        train = dataloader.loadtrain()
        trainval = train + dataloader.loadval() # 训练集 + 验证集
        for sample in trainval:
            data = dataloader.ldintensinfo(sample, dark = True, stand = True)
            data = pretreat.execute(data) # 光谱预处理
            elm.loaddata(data)
        elm.fitpara()
        
        elm.clear()
        # predict
        test = dataloader.loadtest()           
        for sample in test:
            data = dataloader.ldintensinfo(sample, dark = True, stand = True)
            data = pretreat.execute(data) # 光谱预处理
            elm.loaddata(data)
        pred, real = elm.format()
    elif args.backbone == 'ann':
        ann = ANNModel(args)
        for sample in test:
            data = dataloader.ldintensinfo(sample, dark = True, stand = True)
            data = pretreat.execute(data) # 光谱预处理
            ann.loaddata(data)
        saver.load(args, ann.getmodel())
        pred, real = ann.predict()
    elif args.backbone == 'elman': # ElmanRNN
        elman = ElmanModel(args)
        for sample in test:
            data = dataloader.ldintensinfo(sample, dark = True, stand = True)
            data = pretreat.execute(data) # 光谱预处理
            elman.loaddata(data)
        saver.load(args, elman.getmodel())
        pred, real = elman.predict()
    elif args.backbone == 'tnet': # TNet
        tnet = TNetModel(args)
        for sample in test:
            data = dataloader.ldintensinfo(sample, dark = True, stand = True)
            data = pretreat.execute(data) # 光谱预处理
            tnet.loaddata(data)
        saver.load(args, tnet.getmodel())
        pred, real = tnet.predict()
 
    # 设置模型输入，包括：通道数，分辨率等
    #dummy_input = torch.randn(7, 1, 256, device='cuda')
    #h0 = torch.randn(1, 1, 64, device='cuda')
    # 转换为ONNX模型
    #torch.onnx.export(elman.model, (dummy_input, h0), "elman.onnx", export_params=True, opset_version=12)

    # 评价指标
    evaluate.execute(pred, real)
    datashow.show_result(pred, real)
    # 回归曲线图
    #datashow.show_rescompare(pred, real)
    print(args)

if __name__ == "__main__":
    main()