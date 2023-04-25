import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10,20),
                      nn.ReLU(),
                      nn.Linear(20,20),
                      nn.ReLU(),
                      nn.Linear(20,1))

temp = model(2*torch.ones(10))
temp.backward()
lamda = 0.5


with torch.no_grad():
    norm = 0
    for param in model[4].parameters():
        print(torch.norm(param.grad))
        norm += torch.norm(param.grad)**2
    norm = torch.sqrt(norm)
    
    for param in model[4].parameters():
        param.grad = param.grad/(max(1,norm/lamda))
  

for param in model[4].parameters():
    print(torch.norm(param.grad))

