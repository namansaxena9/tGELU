import numpy as np
import torch
import torch.nn as nn
from model import Agent
from buffer import BufferList
import sys
from logger import save_parameters
from config import config
import pickle
from CartPoleEnv import CartPoleEnv
import gym

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

env = CartPoleEnv()
env_eval = CartPoleEnv()

config['state_dim'] = env.observation_space.shape[0]
config["action_dim"] = env.action_space.n

config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

buffer = BufferList(config['buffer_size'])
agent = Agent(env_eval, config)
total_env_steps = config['total_env_steps']
batch_size = config['batch_size']
learning_start = config['warmup_samples']
eval_freq = config['eval_freq']
update_freq = config['update_freq']


epi_len = config['epi_len']


n_steps = 0
count = 0

print("self Q-learning training...")

while(count < total_env_steps//eval_freq):
    state, _ = env.reset()
    done = False
    while(not done):
        n_steps += 1
        
        #if(epi_steps == epi_len):
        #     done  = True  

        if(len(buffer)<learning_start):
            action = np.random.randint(0,config['action_dim'])
        else:
            action = agent.get_action(torch.tensor(state).float()).detach()
            if(n_steps <= config['anneal_limit']):
                agent.anneal()
                
        next_state, reward, done, _, _ = env.step(int(action))
               
        buffer.insert(state,reward,action,done,next_state)
                     
        state = next_state        
        
        if(n_steps % eval_freq == 0):
          count+=1    
          temp = agent.evaluate(epi_len = config['epi_len_eval'], n_iter = config['n_iter_eval'])
          
          agent.logger.add_scalar("Discounted_Reward",temp[2])
          agent.logger.add_scalar('Reward',temp[0])
          agent.logger.add_scalar('rho_eval', temp[1])
          
          save_parameters(agent,config['log_dir'])
          
          print(count," Reward", temp[0], flush = True)
           
    
        if(len(buffer)>=batch_size and n_steps % update_freq == 0):
            agent.update(buffer, batch_size = batch_size, gradient_steps = update_freq, q_freq = config['q_freq'])

agent.logger.flush()        