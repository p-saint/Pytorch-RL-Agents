from commons.utils import NormalizedActions
from gym.wrappers import FlattenObservation
from commons.run_expe import load_config
from commons.network_modules import CriticNetwork, CriticNetworkProgressive
import gym
import torch
device = torch.device('cuda')
config = load_config('results/PNNSAC/LunarLanderContinuous/config.yaml')
eval_env = NormalizedActions(FlattenObservation(gym.make(**config['GAME'])))
state_size = eval_env.observation_space.shape[0]
action_size = eval_env.action_space.shape[0]
cn = CriticNetwork(state_size,action_size,config['HIDDEN_Q_LAYERS'])
cnp = CriticNetworkProgressive(state_size,action_size,config['HIDDEN_Q_LAYERS']).to(device)
filename = 'results/PNNSAC/LunarLanderContinuous_prog_c/models/soft_Q.pth'
print(cnp.state_dict())
cnp.load(filename,device,True)
