import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn

model = nn.Linear(10,1)
optim = SGD(model.parameters(),lr=1)
lr_schedule = LambdaLR(optim, lambda epoch: 1/(epoch**(0.5)+1))


for i in range(1000000):
  if(i%1000==0):  
    print(optim.param_groups[0]['lr'])
  lr_schedule.step()
  