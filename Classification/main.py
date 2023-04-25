import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import os
import random
from model import CNNClassifier
from model import ImageData
from TGeLU import TGeLU
from torch.optim import Adam, SGD
import numpy as np

config = {}
config['seed'] = 123

torch.manual_seed(config['seed'])
torch.cuda.manual_seed(config['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


config['lr'] = 3e-3
config['log_freq'] = 10
config['log_dir'] = 'exp17'
config['train_path'] = './dataset/training_set/'
config['test_path'] = './dataset/test_set/'
config['img_size'] = (64,64)
config['n_iter'] = 400
config['device'] = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
config['tl'] = -1
config['tr'] = 3
config['optim'] = SGD
config['act_fn'] = TGeLU(config['tl'], config['tr'], config['device'])
#config['act_fn'] = nn.GELU()
config['batch_size'] = 256
config['lambda'] = 1

print("Configuration::",config)

model = CNNClassifier(config)

model.train(config['train_path'], config['n_iter'])
print(model.test(config['test_path']))
model.logger.flush()
"""
dataset  = ImageData(config['train_path'], config['img_size'])
data = DataLoader(dataset, batch_size=256)

for i,batch in enumerate(data):
    print(batch[0])
    print(batch[1])
    break
"""    