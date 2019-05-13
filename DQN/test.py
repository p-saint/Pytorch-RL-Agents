import sys
sys.path.extend(['../commons/'])

import argparse
import yaml
import gym
import torch

from model import Model
from utils import get_latest_dir

print("Testing DQN...")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

default_dir = 'runs/' + get_latest_dir('runs/')
parser = argparse.ArgumentParser(description='Test DDPG on ' + config["GAME"])
parser.add_argument('--render', action='store_true', dest="render",
                    help='Display the tests')
parser.add_argument('-n', '--nb_tests', default=10, type=int, dest="nb_tests",
                    help="Number of evaluation to perform.")
parser.add_argument('-f', '--folder', default=default_dir, type=str, dest="folder",
                    help="Folder where the models are saved")
parser.add_argument('--gpu', action='store_true', help='Use GPU')
args = parser.parse_args()


device = torch.device('cpu')

# Create gym environment
env = gym.make(config["GAME"])

STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n

# Creating neural networks and loading models
model = Model(device, STATE_SIZE, ACTION_SIZE, args.folder, config)
model.load()

# START EVALUATION

try:
    rewards = []

    for ep in range(args.nb_tests):

        state = env.reset()
        state = torch.tensor([state], dtype=torch.float)
        done = False
        reward = 0

        while not done:

            action = model.select_action(state, 0, evaluation=True)
            state, r, done, _ = env.step(action)
            if args.render:
                env.render()
            state = torch.tensor([state], dtype=torch.float)
            reward += r

        rewards.append(reward)

except KeyboardInterrupt:
    pass

finally:
    env.close()

print("Average reward over", args.nb_tests, " episodes : ", sum(rewards) / args.nb_tests)
