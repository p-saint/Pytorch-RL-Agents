from commons.utils import NormalizedActions
from gym.wrappers import FlattenObservation
from commons.run_expe import load_config
from commons.network_modules import CriticNetwork, CriticNetworkProgressive
import gym
import torch
device = torch.device('cuda')
config = load_config('results/SAC/LunarLanderContinuous_2020-01-21_10-54-44/config.yaml')
eval_env = NormalizedActions(FlattenObservation(gym.make(**config['GAME'])))
state_size = eval_env.observation_space.shape[0]
action_size = eval_env.action_space.shape[0]
cn = CriticNetwork(state_size,action_size,config['HIDDEN_Q_LAYERS'])
cnp = CriticNetworkProgressive(state_size,action_size,config['HIDDEN_Q_LAYERS'])

path = 'results/SAC/LunarLanderContinuous_2020-01-21_10-54-44/models/soft_Q.pth'
print('CN avant loading: ')
print(cn)

#cn.load(path,device)
print('CN après loading: ')
print(cn)

print('CNP avant loading: ')
print(cnp)

cnp.load(path,device)
print('CNP après loading: ')
print(cnp)

print('New task: ')
cnp.new_task()
print(cnp)
