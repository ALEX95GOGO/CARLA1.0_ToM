import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as scio
import os
import pygame
import signal
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from torch.utils.tensorboard import SummaryWriter   
writer = SummaryWriter('./algo/checkpoints/log')

from env_multi_agent_car_following import CarFollowing
from utils import set_seed, signal_handler

directory = r'./ddpg_save/'
# Arguments
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--algorithm', type=int, help='RL algorithm (0 for Proposed, 1 for IARL, 2 for HIRL, 3 for Vanilla TD3) (default: 0)', default=0)
parser.add_argument('--human_model', action="store_true", help='whehther to use human behavior model (default: False)', default=False)
parser.add_argument('--human_model_update', action="store_true", help='whehther to update human behavior model (default: False)', default=False)
parser.add_argument('--maximum_episode', type=float, help='maximum training episode number (default:400)', default=400)
#parser.add_argument('--seed', type=int, help='fix random seed', default=2)
parser.add_argument("--initial_exploration_rate", type=float, help="initial explore policy variance (default: 0.5)", default=0.5)
parser.add_argument("--cutoff_exploration_rate", type=float, help="minimum explore policy variance (default: 0.05)", default=0.05)
parser.add_argument("--exploration_decay_rate", type=float, help="decay factor of explore policy variance (default: 0.99988)", default=0.99988)
parser.add_argument('--resume', action="store_true", help='whether to resume trained agents (default: False)', default=False)
parser.add_argument('--warmup', action="store_true", help='whether to start training until collecting enough data (default: False)', default=False)
parser.add_argument('--warmup_threshold', type=int, help='warmup length by step (default: 1e4)', default=1e4)
parser.add_argument('--reward_shaping', type=int, help='reward shaping scheme (0: none; 1:proposed) (default: 1)', default=1)
parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda:0')
parser.add_argument('--simulator_port', type=int, help='Carla port value which needs specifize when using multiple CARLA clients (default: 2000)', default=2000)
parser.add_argument('--simulator_render_frequency', type=int, help='Carla rendering frequenze, Hz (default: 25)', default=25)
parser.add_argument('--simulator_conservative_surrounding', action="store_true", help='surrounding vehicles are conservative or not (default: False)', default=False)
parser.add_argument('--joystick_enabled', action="store_true", help='whether use Logitech G29 joystick for human guidance (default: False)', default=False)

parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=10, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=1, type=int)
parser.add_argument('--mode', type=str, help='choose: auto, manual, steer, pedal', default='auto')
parser.add_argument('--town', type=int, help='choose: 4,7', default='4')

args = parser.parse_args()

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(args.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(args.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(args.device)
        self.critic_target = Critic(state_dim, action_dim).to(args.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(args.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(args.device)
            action = torch.FloatTensor(u).to(args.device)
            next_state = torch.FloatTensor(y).to(args.device)
            done = torch.FloatTensor(1-d).to(args.device)
            reward = torch.FloatTensor(r).to(args.device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self,i):
        torch.save(self.actor.state_dict(), directory + 'actor.pth'.format(i))
        torch.save(self.critic.state_dict(), directory + 'critic.pth'.format(i))
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self,i):
        self.actor.load_state_dict(torch.load(directory + 'actor{}.pth'.format(i)))
        self.critic.load_state_dict(torch.load(directory + 'critic{}.pth'.format(i)))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def train_leftturn_task():
    
    set_seed(args.seed)
    

    env = CarFollowing(joystick_enabled = args.joystick_enabled, conservative_surrounding = args.simulator_conservative_surrounding, 
                   frame=args.simulator_render_frequency, port=args.simulator_port, mode=args.mode, town=args.town)

    s_dim = [env.observation_size_width, env.observation_size_height]
    a_dim = env.action_size
    s_dim = 3
    a_dim = 1

    max_action = 1
    #DRL = DRL(a_dim, s_dim, device=args.device)
    agent = DDPG(s_dim, a_dim, max_action)
    agent.load('_best')
    exploration_rate = args.initial_exploration_rate 
    
    if args.resume and os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        agent.load(log_dir)
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    
    
    # initialize global variables
    total_step = 0
    a_loss,c_loss = 0,0
    count_sl = 0
    
    loss_critic, loss_actor = [], []
    episode_total_reward_list, episode_mean_reward_list = [], []
    global_reward_list, episode_duration_list = [], []
    args.maximum_episode = int(args.maximum_episode)
    drl_action = [[] for i in range(args.maximum_episode)] # drl_action: output by RL policy
    adopted_action = [[] for i in range(args.maximum_episode)]  # adopted_action: eventual action which may include human guidance action
     
    reward_i_record = [[] for i in range(args.maximum_episode)]  # reward_i: virtual reward (added shaping term);
    reward_e_record = [[] for i in range(args.maximum_episode)]  # reward_e: real reward
    
    
    start_time = time.perf_counter()
    
    for i in range(start_epoch, args.maximum_episode):
        
        # initial episode variables
        ep_reward = 0      
        step = 0
        step_intervened = 0
        done = False
        human_model_activated = False

        # initial environment
        observation = env.reset()
        #state = np.repeat(np.expand_dims(observation,2), 2, axis=2)
        #state = np.repeat(np.expand_dims(observation,1), 2, axis=1)
        state = observation
        state_ = state.copy()
        #import pdb; pdb.set_trace()
    
        while not done:
            ## Section DRL's actting ##
            #action = DRL.choose_action(state[:,1])
            #action = np.clip( np.random.normal(action,  exploration_rate), -1, 1)
            action = agent.select_action(state)

            drl_action[i].append(action)
            
            adopted_action[i].append(action)

            #state = state_.copy()
            observation_, action_fdbk, reward_e, _, done, scope = env.step(action)
            observation_ = np.array(observation_)
            state = observation_
            dura = env.terminate_position
            total_step += 1
            step += 1
            exploration_rate = exploration_rate * args.exploration_decay_rate if exploration_rate>args.cutoff_exploration_rate else args.cutoff_exploration_rate
            signal.signal(signal.SIGINT, signal_handler)
        

        mean_reward =  ep_reward / step  
        episode_total_reward_list.append(ep_reward)
        episode_mean_reward_list.append(mean_reward)
        episode_duration_list.append(dura)

        print('\n episode is:',i)
        #print('explore_rate:',round(exploration_rate,4))
        #print('c_loss:',round(np.average(c_loss),4))
        #print('a_loss',round(np.average(a_loss),4))
        print('total_step:',total_step)
        print('episode_step:',step)
        print('episode_cumu_reward',round(ep_reward,4))
        #import pdb; pdb.set_trace()
        writer.add_scalar('reward/reward_episode', ep_reward, i)
        writer.add_scalar('reward/reward_episode_noshaping', np.mean(reward_e_record[i]), i)
        writer.add_scalar('reward/duration_episode', step, i)
        #writer.add_scalar('rate_exploration', round(exploration_rate,4), i)
        #writer.add_scalar('loss/loss_critic', round(np.average(c_loss),4), i)
        #writer.add_scalar('loss/loss_actor', round(np.average(a_loss),4), i)
        
        
    print('total time:',time.perf_counter()-start_time)        
    
    timeend = round(time.time())
    #DRL.save_actor('./algo/models_leftturn', timeend)
    
    pygame.display.quit()
    pygame.quit()
    
    action_drl = drl_action[0:i]
    action_final = adopted_action[0:i]
    scio.savemat('dataleftturn_{}-{}-{}.mat'.format(args.seed,args.algorithm,timeend), mdict={'action_drl': action_drl,'action_final': action_final,
                                                    'stepreward':global_reward_list,'mreward':episode_mean_reward_list,
                                                    'step':episode_duration_list,'reward':episode_total_reward_list,
                                                    'r_i':reward_i_record,'r_e':reward_e_record})
    
    


if __name__ == "__main__":


    # Run
    train_leftturn_task()
