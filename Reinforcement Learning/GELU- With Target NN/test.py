import numpy as np
import torch
import torch.nn as nn
from PendulumEnv import PendulumEnv
from model import Agent
from buffer import BufferList
import sys
from logger import load_parameters
from config import config
import matplotlib.pyplot as plt

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

env_eval = PendulumEnv(config)

config['state_dim'] = env_eval.get_state_dim()
config["action_dim"] = env_eval.get_action_dim()
config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


buffer = BufferList(config['buffer_size'])
agent = Agent(env_eval, config)

load_parameters(agent,'log',config['device'])
epi_len = config['epi_len_eval']

plot = []
n_iter = 1
for epi in range(n_iter):
    total_reward = 0
    state = env_eval.reset()
    done = False
    steps = 0    
    while not done:
        action = agent.get_action(torch.tensor(state).float()).detach()
        next_state, reward, done, _ = env_eval.step(int(action))
        print("action",action)
        plot.append(state)
        total_reward += reward
        state = next_state
        steps +=1
    print("Reward",total_reward)
    print("steps",steps)

plot = np.array(plot)
plt.plot(plot[:,0])
plt.plot(plot[:,1])

