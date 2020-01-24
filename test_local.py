from commons.utils import NormalizedActions
from gym.wrappers import FlattenObservation
from commons.run_expe import load_config
from commons.network_modules import CriticNetwork, CriticNetworkProgressive
import gym
import torch
# device = torch.device('cuda')
# config = load_config('results/PNNSAC/LunarLanderContinuous/config.yaml')
# eval_env = NormalizedActions(FlattenObservation(gym.make(**config['GAME'])))
# state_size = eval_env.observation_space.shape[0]
# action_size = eval_env.action_space.shape[0]
# cn = CriticNetwork(state_size,action_size,config['HIDDEN_Q_LAYERS'])
# cnp = CriticNetworkProgressive(state_size,action_size,config['HIDDEN_Q_LAYERS']).to(device)
# filename = 'results/PNNSAC/LunarLanderContinuous_prog_c/models/soft_Q.pth'
# print(cnp.state_dict())
# cnp.load(filename,device,True)


#gym.register(id='HandManipulateBlockCustom-v0',entry_point='gym.envs.robotics:HandBlockEnv',kwargs={'target_position': 'random', 'target_rotation': 'xyz','reward_type': 'dense'},max_episode_steps=500)

params = {'id': 'HandManipulateBlockCustom-v0', 'entry_point': 'gym.envs.robotics:HandBlockEnv',
'kwargs': {'target_position': 'random', 'target_rotation': 'xyz','reward_type': 'dense'},
'max_episode_steps': 2}
gym.register(**params)
env = gym.make(params['id'])
print(dir(env))
done = False
env.reset()
steps = 0
while not done:
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    print('Reward: {}'.format(reward))
    steps += 1
print(env.spec.max_episode_steps)
print('Number of steps: ',steps)
