import joblib
import torch
from netmodel.mlre import *

class ModelSaver:
    def __init__(self):
        self = self

    def save(self, args, model):
        if args.backbone == 'slr':
            joblib.dump(model, 'models/slr.pkl')
        elif args.backbone == 'plsr':
            joblib.dump(model, 'models/plsr.pkl')
        elif args.backbone == 'lssvr':
            joblib.dump(model, 'models/lssvr.pkl')
        elif args.backbone == 'ann':
            torch.save(model.state_dict(), 'models/ann.pth')
        elif args.backbone == 'elman':
            torch.save(model.state_dict(), 'models/elman.pth')
        elif args.backbone == 'tnet':
            torch.save(model.state_dict(), 'models/tnet.pth')

    def load(self, args, model):
        if args.backbone == 'slr':
            model = joblib.load('models/slr.pkl')
        elif args.backbone == 'plsr':
            model = joblib.load('models/plsr.pkl')
        elif args.backbone == 'lssvr':
            model = joblib.load('models/lssvr.pkl')
        elif args.backbone == 'ann':
            model.load_state_dict(torch.load('models/ann.pth'))
        elif args.backbone == 'elman':
            model.load_state_dict(torch.load('models/elman.pth'))
        elif args.backbone == 'tnet':
            model.load_state_dict(torch.load('models/tnet.pth'))
        return model