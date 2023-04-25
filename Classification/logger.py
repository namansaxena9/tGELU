#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import os
import numpy as np
import torch
import pickle


def save_parameters(model,path):
    torch.save(model.state_dict(),'./'+path+'/model.pt')

def load_parameters(model,path):
    model.load_state_dict(torch.load('./'+path+'/model.pt',map_location=torch.device('cpu')))


class Logger(object):
    def __init__(self, seed, log_dir = None, log_freq = 10):
        self.data = {}
        if(log_dir is None):
           self.log_dir = './log'
        else:
           self.log_dir = log_dir
        
        if(not os.path.isdir(self.log_dir)):
            os.mkdir(self.log_dir)
        
        self.seed = seed
        self.pointer = 0
        self.log_freq = log_freq

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def add_scalar(self, tag, value, step=None):
        value = self.to_numpy(value)
        if(tag in self.data):
            self.data[tag].append(value)
        else:
            self.data[tag] = []
            self.data[tag].append(value)

        self.pointer +=1
        if(self.pointer % self.log_freq == 0):
            pointer = 0
            self.flush()

    def flush(self):
        file = open(self.log_dir + '/'+str(self.seed)+'.pkl','wb')
        pickle.dump(self.data, file)
        file.close()


