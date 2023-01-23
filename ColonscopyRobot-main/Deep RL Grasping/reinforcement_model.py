import os
import random
import numpy as np

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# CUDA Device Selector
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Torch Flag to detect any autograd errors
torch.autograd.set_detect_anomaly(True)

# Random Seed
random.seed(random.randint(0, 10000))

# Path to save network models
PATH_ACTOR_MODELS = os.path.join(os.getcwd(), "models/actor")
PATH_CRITIC_MODELS = os.path.join(os.getcwd(), "models/critic")
PATH_VALUE_MODELS = os.path.join(os.getcwd(), "models/value")


class ActorNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, beta):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(n_inputs + n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.q(x)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, n_inputs, beta):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.v(x))
        return x

class CriticNetwork(nn.Module):
    def __init__(self, alpha, n_input, n_actions, max_action):
        super(CriticNetwork, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(n_input, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, n_actions)
        self.sigma = nn.Linear(256, n_actions)
        self.reparam_noise = 1e-6
        self.alpha = alpha
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # Limit sigma between [reparam, 1]
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma
    
    def sample(self, state, reparam=True):
        mu, sigma = self.forward(state)
        
        # a = tanh(n)   where n = Norm(mu_pi, sigma_pi)
        # log(pi(a | s)) = log(êta(n | s)) - sum(log(1 - tanh(n_i)^2))
        
        # Take the neural output and create normal distrubution to sample actions
        # return the log(pi(a | s)) where a = Norm(mu_pi, sigma_pi)

        # create Normal distribuion from forward pass

        prob = Normal(mu, sigma)
        
        if reparam:
            a_n = prob.rsample()
        else:
            a_n = prob.sample()

        a = torch.tanh(a_n) * torch.Tensor(self.max_action).to(device)
        log_a = prob.log_prob(a_n)
        log_a -= torch.log(1-a.pow(2))
        log_a = log_a.sum(1, keepdim=True)
        
        return a, log_a

class ReplayBuffer:
    def __init__(self, batch, max_buffer):
        self.__batch_size = batch               # Batch Size of Experiences
        self.__max_buffer = max_buffer          # Max Size of Memory Buffer
        self.__memory = []                      # Memory Queue
    
    def add(self, action, state, reward, next_state, done):
        if len(self.__memory) > self.__max_buffer:
            self.__memory.pop(0)
        self.__memory.append([action, state, reward, next_state, done])     # Memory buffer has the following structure [[[action], [state], [reward], [next_state], [done]],
                                                                            #											 [[action], [state], [reward], [next_state], [done]],
																		    #											 ....
																			#											 [[action], [state], [reward], [next_state], [state]]]
																			#
																			# The class buffer as list but we convert to numpy array and finally a torch array for backward network pass

    def sample(self):
        if self.__batch_size > len(self.__memory):
            return None, None, None, None, None

        sample_index = [random.randint(0, len(self.__memory)-1) for i  in range(self.__batch_size)]
        actions = [self.__memory[random_num][0] for random_num in sample_index]
        states = [self.__memory[random_num][1] for random_num in sample_index]
        rewards = [self.__memory[random_num][2] for random_num in sample_index]
        next_states = [self.__memory[random_num][3] for random_num in sample_index]
        dones = [self.__memory[random_num][4] for random_num in sample_index]
        torch_actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        torch_states = torch.from_numpy(np.vstack(states)).float().to(device)
        torch_rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        torch_next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        torch_dones = torch.from_numpy(np.vstack(dones)).int().to(device)
        return (torch_actions, torch_states, torch_rewards, torch_next_states, torch_dones)

    def __len__(self):
        return len(self.__memory)


class Agent:
    def __init__(self, batch_size, max_buffer_size, num_inputs, num_actions, max_action
                        , scale=2, alpha=0.0003, beta=0.0003, gamma=0.99, tau=0.005):
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__tau = tau

        # replay buffer
        self.__memory = ReplayBuffer(batch_size, max_buffer_size)
        self.__batch = batch_size
        self.__max_buffer = max_buffer_size

        # initialize networks
        self.actor1 = ActorNetwork(num_inputs, num_actions, beta)                       # Q-Network 1
        self.actor2 = ActorNetwork(num_inputs, num_actions, beta)                       # Q-Network 2
        self.critic = CriticNetwork(alpha, num_inputs, num_actions, max_action)         # Policy-Network
        self.value = ValueNetwork(num_inputs, beta)                                             
        self.value_target = ValueNetwork(num_inputs, beta)

        self.scale = scale
        # copy of value functions networks
        self.__soft_update(tau=1)
    
    def step(self, action, state, reward, next_step, done):
        self.__memory.add(action, state, reward, next_step, done)

        if len(self.__memory) > self.__batch:
            experience = self.__memory.sample()
            self.__learn(experience)

    def act(self, observation):
        tensorState = torch.Tensor([observation]).to(device)
        actions, _ = self.critic.sample(tensorState, reparam=False)
        action = actions.cpu().detach().numpy()[0]
        return action

    def save_models(self):
        torch.save(self.actor1.state_dict(), os.path.join(PATH_ACTOR_MODELS, "actor1_network.pth"))
        torch.save(self.actor1.state_dict(), os.path.join(PATH_ACTOR_MODELS, "actor1_network.pth"))
        torch.save(self.critic.state_dict(), os.path.join(PATH_CRITIC_MODELS, "crtic_network.pth"))
        torch.save(self.value.state_dict(), os.path.join(PATH_VALUE_MODELS, "value_local_network.pth"))
        torch.save(self.value_target.state_dict(), os.path.join(PATH_VALUE_MODELS, "value_target_network.pth"))

    def load_models(self):
        self.actor1.load_state_dict(torch.load(os.path.join(PATH_ACTOR_MODELS, "actor1_network.pth"), map_location=device))
        self.actor2.load_state_dict(torch.load(os.path.join(PATH_ACTOR_MODELS, "actor2_network.pth"), map_location=device))
        self.critic.load_state_dict(torch.load(os.path.join(PATH_CRITIC_MODELS, "critic_network.pth"), map_location=device))
        self.value.load_state_dict(torch.load(os.path.join(PATH_VALUE_MODELS, "value_local_network.pth"), map_location=device))
        self.value_target.load_state_dict(torch.load(os.path.join(PATH_VALUE_MODELS, "value_target_network.pth"), map_location=device))

    def __learn(self, experience):
        action, state, reward, next_state, done = experience

        pred_q_value_1 = self.actor1(state, action)
        pred_q_value_2 = self.actor2(state, action)
        target_value = self.value_target(next_state)
        # Use reparameterization, to make sure that sampling from the policy
        # is  a differentiable process so that there are no problems in backpropagating
        # the errors.
        pred_new_action, log_prob = self.critic.sample(state, reparam=True)

        ############# Training Q Function ####################
        # Calculate the Q loss function
        target_q_value = reward + (1 - done) * self.__gamma * target_value
        q_value_loss1 = F.mse_loss(pred_q_value_1, target_q_value.detach())
        q_value_loss2 = F.mse_loss(pred_q_value_2, target_q_value.detach())
        # Using losses perform Q Loss Backprogatation 
        self.actor1.optimizer.zero_grad()
        q_value_loss1.backward()
        self.actor1.optimizer.step()
        self.actor2.optimizer.zero_grad()
        q_value_loss2.backward()
        self.actor2.optimizer.step()

        ############ Training Value Function #################
        # Calculate the V loss function
        pred_value = self.value(state)
        new_q_value_1 = self.actor1(state, pred_new_action)
        new_q_value_2 = self.actor2(state, pred_new_action)
        new_q_value = torch.min(new_q_value_1, new_q_value_2)
        target_value_func = new_q_value - log_prob
        value_loss = F.mse_loss(pred_value, target_value_func.detach())
        # Using losses perform V Loss backprogatation
        self.value.optimizer.zero_grad()
        value_loss.backward()
        self.value.optimizer.step()

        ########### Training Policy Function ################
        # Calculate the Pi Loss Function
        policy_loss = (log_prob - new_q_value).mean()
        # Using losses perform Pi Loss backprogatation
        self.critic.optimizer.zero_grad()
        policy_loss.backward()
        self.critic.optimizer.step()
        print("################# Training Models ####################")

        self.__soft_update()


    def __soft_update(self, tau=None):
        if tau is None:
            tau = self.__tau
        
        for target_param, param in zip(self.value_target.parameters(), self.value.parameters()):
            target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
