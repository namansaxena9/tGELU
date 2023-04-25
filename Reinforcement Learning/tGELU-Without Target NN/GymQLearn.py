import gym
import numpy as np

class GymQLearn(gym.Wrapper):
    def __init__(self, env, config, epi_len):
        super().__init__(env)
        self.action_dict = {}
        self.action_dict_inv = {}

        config['action_min'] = env.action_space.low[0]
        config['action_max'] = env.action_space.high[0]

        for index,i in enumerate(np.arange(config['action_min']*10**config['action_dis'],config['action_max']*10**config['action_dis']+1)):
            i_ = round(i/(10**config['action_dis']),config['action_dis'])
            self.action_dict[i_] = index
            self.action_dict_inv[index] = i_

        self.epi_len = epi_len
        self.action_dim = len(self.action_dict) 

    def step(self, action):        
        state, reward, done, info = self.env.step([self.action_dict_inv[action]])
        return state, reward, done, info
            
        
        