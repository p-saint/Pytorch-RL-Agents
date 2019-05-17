import os
import time
from datetime import datetime
import argparse
try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

import matplotlib.pyplot as plt
import gym
import yaml

import torch
from tensorboardX import SummaryWriter

from model import Model

import utils

print("DQN starting...")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Run DQN on ' + config["GAME"])
parser.add_argument('--no_gpu', action='store_false', dest='gpu', help="Don't use GPU")
args = parser.parse_args()

# Create folder and writer to write tensorboard values
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
folder = f'runs/{config["GAME"].split("-")[0]}_{current_time}'
writer = SummaryWriter(folder)
if not os.path.exists(folder+'/models/'):
    os.mkdir(folder+'/models/')

# Write optional info about the experiment to tensorboard
for k, v in config.items():
    writer.add_text('Config', str(k) + ' : ' + str(v), 0)

with open(folder+'/config.yaml', 'w') as file:
    yaml.dump(config, file)

# Choose device cpu or cuda if a gpu is available
if args.gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("\033[91m\033[1mDevice : ", device.upper(), "\033[0m")
writer.add_text('Device', device, 0)
device = torch.device(device)


# Create gym environment
print("Creating environment...")
env = gym.make(config["GAME"])

STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n


def train():

    print("Creating neural networks and optimizers...")
    model = Model(device, STATE_SIZE, ACTION_SIZE, folder, config)

    nb_total_steps = 0
    time_beginning = time.time()

    try:
        print("Starting training...")
        nb_episodes_done = 0
        rewards = []

        for episode in trange(config['MAX_EPISODES']):

            # Initialize the environment and state
            state = env.reset()
            episode_reward = 0
            done = False
            step = 0

            while step <= config['MAX_STEPS'] and not done:

                # Select and perform an action
                action = model.select_action(state, episode)
                next_state, reward, done, _ = env.step(action)
                reward = model.intermediate_reward(reward, next_state)
                episode_reward += reward.item()

                if not done and step == config['MAX_STEPS']-1:
                    done = True

                # Store the transition in memory
                model.memory.push(state, action, reward, next_state, 1-int(done))
                state = next_state

                loss = model.optimize()

                step += 1
                nb_total_steps += 1

            rewards.append(episode_reward)

            # Update the target network
            if episode % config['TARGET_UPDATE'] == 0:
                utils.update_targets(model.agent.target_nn, model.agent.nn, model.config['TAU'])

            nb_episodes_done += 1

            # Write scalars to tensorboard
            writer.add_scalar('reward_per_episode', episode_reward, episode)
            writer.add_scalar('steps_per_episode', step, episode)
            if loss is not None:
                writer.add_scalar('loss', loss, episode)

            # Stores .png of the reward graph
            if nb_episodes_done % config["FREQ_PLOT"] == 0:
                plt.cla()
                plt.plot(rewards)
                plt.savefig(folder+'/rewards.png')

    except KeyboardInterrupt:
        pass

    finally:
        env.close()
        writer.close()
        model.save()
        print("\n\033[91m\033[1mModel saved in", folder, "\033[0m")

    time_execution = time.time() - time_beginning

    print('\n---------------------STATS-------------------------\n',
          nb_total_steps, ' steps and updates of the network done\n',
          nb_episodes_done, ' episodes done\n'
          'Execution time : ', round(time_execution, 2), ' seconds\n'
          'Average nb of steps per second : ', round(nb_total_steps/time_execution, 3), 'steps/s\n'
          'Average duration of one episode : ', round(time_execution/nb_episodes_done, 3), 's\n')


if __name__ == '__main__':
    train()
