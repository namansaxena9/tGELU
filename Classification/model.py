import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader
import os
import random
import numpy as np
from logger import Logger

class ImageData(Dataset):
    def __init__(self, path, size):
        super(ImageData,self).__init__()
        self.path = path
        self.size = size
        self.files = os.listdir(self.path)
        random.shuffle(self.files)
        
    def __getitem__(self, index):
        img = cv2.imread(self.path + self.files[index],cv2.IMREAD_GRAYSCALE)
        if(self.files[index][0] == 'c'):
            class1 = torch.LongTensor([0])
        else:
            class1 = torch.LongTensor([1])
        return ((torch.FloatTensor(cv2.resize(img, self.size))-128)/128).unsqueeze(0), class1
    
    def __len__(self):
        return len(self.files)
    

class CNNClassifier(nn.Module):
    def __init__(self,config):
        super(CNNClassifier,self).__init__()
        self.config = config
        self.device = self.config['device']
        
        torch.manual_seed(self.config['seed'])
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        self.logger = Logger(self.config['seed'], self.config['log_dir'], self.config['log_freq'])

        self.feature = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1),   
                              self.config['act_fn'],
                              nn.MaxPool2d(2),
                              nn.Conv2d(8, 16, 3, 1, 1),
                              self.config['act_fn'],
                              nn.MaxPool2d(2),
                              nn.Conv2d(16, 64, 3, 1, 1),   
                              self.config['act_fn'],
                              nn.MaxPool2d(2),
                              )
        
        self.feature.to(self.device)

        self.net = nn.Sequential(nn.Linear(64*8*8, 1024),
                                 self.config['act_fn'],
                                 nn.Linear(1024, 256),
                                 self.config['act_fn'],
                                 nn.Linear(256, 2),
                                 nn.Softmax(dim = -1))
        
        self.net.to(self.device)       
        self.optim = self.config['optim'](self.parameters(), lr = self.config['lr'])
    
    def forward(self, in1):
        temp = self.feature(in1)
        temp2 = torch.flatten(temp,start_dim = 1)
        return self.net(temp2)
    
    def reduce_grad(self): 
        with torch.no_grad():
            norm = 0
            for param in self.net[4].parameters():
                norm += torch.norm(param.grad)**2
            norm = torch.sqrt(norm)

            for param in self.net[4].parameters():
                param.grad = param.grad/(max(1,norm/self.config['lambda']))
         
    def train(self, path, n_iter):
        data = ImageData(path, self.config['img_size'])
        dataset = DataLoader(data, batch_size = self.config['batch_size'])
        loss_fn = nn.CrossEntropyLoss()
        
        for iter1 in range(n_iter):
            total_loss = 0
            total_samples = 0
            accuracy = 0             
            for i, batch in enumerate(dataset):
                x_train = batch[0].to(self.device)
                y_train = batch[1].reshape(-1).to(self.device)
                total_samples+=len(x_train)
                #print(sum(y_train)/len(y_train))
                out1 = self(x_train)
                loss = loss_fn(out1,y_train)
                total_loss +=loss.item()
                self.optim.zero_grad()
                loss.backward()
                if(self.config['lambda']!=None):
                   self.reduce_grad()
                self.optim.step()
                with torch.no_grad():
                    predict = torch.argmax(out1,dim = 1).to('cpu')
                    predict = np.array(predict, dtype = 'int')
                    y_train = np.array(y_train.to('cpu'), dtype='int').reshape(-1) 
                    accuracy+=sum(predict==y_train)
            #print("Loss:",total_loss)
            print("Accuracy: ", accuracy/total_samples, flush = True)
            self.logger.add_scalar("train_accuracy", accuracy/total_samples)
            if(iter1 % 20 ==0):
                  self.logger.add_scalar("test_accuracy", self.test(self.config['test_path'])) 
    
    def test(self, path):
        data = ImageData(path, self.config['img_size'])
        dataset = DataLoader(data, batch_size = len(data))
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                x_test = batch[0].to(self.device)
                y_test = batch[1]                
                logits = self(x_test)
                predict = torch.argmax(logits,dim = 1).to('cpu')
                predict = np.array(predict, dtype = 'int')
                y_test = np.array(y_test, dtype='int').reshape(-1)
        return sum(predict==y_test)/len(y_test)

            
                       
        
        
        
                            
