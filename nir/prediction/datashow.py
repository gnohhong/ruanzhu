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

def main():
    dataloader = Dataloader()
    _size, _wave, _dark = dataloader.get_baseinfo() # 采样点，波长，暗底
    _stand = dataloader.get_standinfo() # 标准光谱
    datashow = Datashow(_wave, _dark, _stand)
    pretreat = Pretreatment(sg=False, snv=False, msc=False, dt=False, maxmin=True, mean=False, 
                                vector=False, d1=False, d2=False, wave=False)
    data = dataloader.ldintensallinfo(dark = True, stand = True)
    data = pretreat.execute(data) # 光谱预处理

    datashow.showdata(data)

if __name__ == "__main__":
    main()