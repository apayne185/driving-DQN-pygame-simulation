import sys, os
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import gymnasium as gym
import gym_race

VERSION_NAME = 'DQN_v01'



class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)
    


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)

        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.epsilon = EPSILON_START
        self.epsilon_decay = (EPSILON_START-EPSILON_END)/EPSILON_DECAY
        self.update_target_every = 500 
    
    

    def train(self, batch_size, step):
        if len(self.memory) < batch_size:
            return 0.0
        
        # transitions = self.memory.sample(batch_size)
        # batch = list(zip(*transitions))
        batch = self.memory.sample(batch_size)
        # state_batch = torch.FloatTensor(batch[0]).to(self.device)
        # action_batch = torch.LongTensor(batch[1]).to(self.device)
        # reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        # next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        # done_batch = torch.FloatTensor(batch[4]).to(self.device)
        state_batch = torch.tensor(batch[0], dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=self.device)
            
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        with torch.no_grad():
            # next_q_values = self.policy_net(next_state_batch).max(1)[0]
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        
        self.epsilon = max(EPSILON_END, self.epsilon-self.epsilon_decay)
        
        return loss.item()
    

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)

            return q_values.max(1)[1].item()
        
        
        
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        batch = list(zip(*batch)) 
        return [np.array(x, dtype=np.float32) for x in batch]

        # return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
        


def simulate(episode_start= 0):
    env = gym.make("Pyrace-v1").unwrapped
    # print("Environment:", type(env))

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # print(f"Input dim: {state_size}, Num actions: {action_size}")

    
    agent = DQNAgent(state_size, action_size)
    total_rewards = []
    max_reward = -10_000
    
    model_dir = f'models_{VERSION_NAME}'
    if not os.path.exists(model_dir): os.makedirs(model_dir)
        
    
    for episode in range(episode_start, NUM_EPISODES + episode_start):
        state, _ = env.reset()
        total_reward = 0
        episode_losses = []
        
        for t in range(MAX_T):
            env.render()
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.memory.push(state, action,reward, next_state,done)
            
            loss = agent.train(BATCH_SIZE, t)      #need to inclued step here
            if loss > 0:
                episode_losses.append(loss)
            
            total_reward += reward
            state = next_state
            
            if episode % DISPLAY_EPISODES == 0:
                env.set_msgs(['SIMULATE',
                            f'Episode: {episode}',
                            f'Time steps: {t}',
                            f'check: {info["check"]}',
                            f'dist: {info["dist"]}',
                            f'crash: {info["crash"]}',
                            f'Reward: {total_reward:.0f}',
                            f'Max Reward: {max_reward:.0f}'])
                # env.render()
            
            if done or t >= MAX_T - 1:
                break
        
        total_rewards.append(total_reward)
        if total_reward > max_reward:
            max_reward = total_reward
            file = os.path.join(model_dir, 'best_dqn_model.pth')
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'total_rewards': total_rewards,
                }, 
            file)
            print(f'New best model saved {file}')

    print("Training complete.")
        
        # if episode % REPORT_EPISODES == 0:
        #     plt.figure()
#             plt.plot(rewards_history)
#             plt.ylabel('Total Reward')
#             plt.xlabel('Episode')    
#             plt.title('DQN Training Performace ')
            # plt.close()

            
            # Save model
            # file = os.path.join(model_dir, f'dqn_model_{episode}.pth')
            # torch.save({
            #     'episode': episode,
            #     'model_state_dict': agent.policy_net.state_dict(),
            #     'optimizer_state_dict': agent.optimizer.state_dict(),
            #     'epsilon': agent.epsilon,
            #     'total_rewards': total_rewards,
            #     }, 
            # file)
            # print(f'model saved to {file}')




def load_and_play(episode,learning= False):
    env = gym.make("Pyrace-v1")#.unwrapped
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    # file = os.path.join(f'models_{VERSION_NAME}', f'dqn_model_{episode}.pth')
    if episode == "best":
        file = os.path.join(f'models_{VERSION_NAME}', 'best_dqn_model.pth')
    else:
        file = os.path.join(f'models_{VERSION_NAME}', f'dqn_model_{episode}.pth')

    if not os.path.exists(file):
        print(f"Error: Model file {file} not found!")
        return

    checkpoint = torch.load(file)

    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    
    if learning:
        simulate(episode_start=0)  
    else:
        state, _ = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.select_action(state)
            state, _, done, _, _ = env.step(action)





if __name__ == "__main__":
    NUM_EPISODES = 15_000
    MAX_T = 2000
    REPORT_EPISODES = 500
    DISPLAY_EPISODES = 100
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 10000
    LEARNING_RATE= 0.001
    MEMORY_SIZE = 10000

    
    # simulate()                            #start with this
    load_and_play("best", learning=True)      #plays the game, can continue training
