import torch
import torch.nn as nn
import numpy as np
from TGeLU import TGeLU
from torch.optim import Adam, SGD

config = {}

config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
config['seed'] = 300
config['buffer_size'] = int(1e6)
config['total_env_steps'] = int(1e6)
config['batch_size'] = 256
config['warmup_samples'] = 5000
config['eval_freq'] = 5000
config['epi_len'] = 1000
config['n_iter_eval'] = 5
config['epi_len_eval'] = 1000
config['act_fn'] = TGeLU(-1, 1, config['device'])
config['tau'] = 0.995
config['log'] = True
config['log_dir'] = 'log'+str(config['seed'])
config['update_freq'] = 1
config['q_freq'] = 1
config['gamma'] = 0.99
config['init_epsilon'] = 1
config['final_epsilon'] = 0.1
config['anneal_limit'] = int(1e5)
config['lambda'] = 1

config['optimizer'] = SGD
config['lr_q'] = 3e-4
config['hidden_layer'] = [128,128]