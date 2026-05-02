import torch
import torch.nn as nn
import numpy as np
import argparse

from utils.dataloader import Dataloader
from utils.chart import Datashow
from utils.saver import ModelSaver
from netmodel.tnet import *
from netmodel.elman import *
from algorism.pretreatment import *

def main():
    parser = argparse.ArgumentParser(description="PyTorch Net Predict....")
    parser.add_argument('--backbone', type=str, default='tnet',
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

    pretreat = Pretreatment(sg=True, snv=False, msc=True, dt=False, maxmin=False, mean=False, 
                                vector=True, d1=False, d2=False, wave=True)

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    
    if args.backbone == 'tnet':
        tnet = TNetModel(args)
        
        data = dataloader.ldintensinfo(40, dark = True, stand = True)
        data = pretreat.execute(data) # 光谱预处理
        _intens, _intime, _humid = zip(*data)
        
        mean = 0
        max = 0
        saver.load(args, tnet.getmodel())
        for i in range(len(_intens)-6):
            intens = _intens[i:i+7]
            intime = _intime[i:i+7]
            humid = _humid
            edata = zip(intens, intime, humid)
            tnet.loaddata(edata)
            
            pred, real = tnet.predict()
            if abs(real - pred) > max:
                max = abs(real - pred)
            mean += abs(pred - real)
            print(pred, real)
            tnet.cleardata()

        mean /= 4
        print(mean, max)

if __name__ == "__main__":
    main()