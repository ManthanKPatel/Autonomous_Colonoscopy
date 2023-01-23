import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Training Parameters
BUFFER = 100				# Replay Buffer Size
BATCH = 32					# Minibatch Size
GAMMA = 0.99				# Discount Factor
TAU = 0.0001				# For Soft Update of Target Parmarter
LR = 0.0001					# Learning RATE
UPDATE = 100				# After num of UPDATE the algorthm will train over Batch of experiences


# CUDA Device Selector
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Random Seed
random.seed(random.randint(0, 1000))		# Random number seed


# Actor (Policy) Model
class QNetwork(nn.Module):
	def __init__(self, states, actions):
		super(QNetwork, self).__init__()
		self.fc1 = nn.Linear(states, 64)
		self.fc2 = nn.Linear(64, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, actions) 

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return self.fc4(x)


# Agent that learns and interacts with env
class Agent:
	def __init__(self, num_states, num_actions):
		self.__num_states = num_states
		self.__num_action = num_actions

		# Q-Networks
		self.qnetwork_local = QNetwork(num_states, num_actions).to(device)
		self.qnetwork_target = QNetwork(num_states, num_actions).to(device)
		self.__optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
		self.__memory = ReplayBuffer(BATCH, BUFFER)
		self.__t_step = 0

	def step(self, action, state, reward, next_step, done):
		self.__memory.add(action, state, reward, next_step, done)

		self.__t_step += 1
		if (self.__t_step % UPDATE == 0):
			if len(self.__memory) > BATCH:
				print("Agent Training on Batch")
				experience = self.__memory.sample()
				self.learn(experience)

	def act(self, state, eps=0):
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.qnetwork_local.eval()
		with torch.no_grad():
			action_values = self.qnetwork_local(state)
		self.qnetwork_local.train()

		# Epsilon - greedy action selection
		if random.random() > eps:
			return np.argmax(action_values.cpu().data.numpy())
		else:
			return random.choice(np.arange(self.__num_action))

	def learn(self, experiences):
		actions, states, rewards, next_states, dones = experiences
		criterion = torch.nn.MSELoss()
		self.qnetwork_local.train()
		self.qnetwork_target.eval()
		predicted_targets = self.qnetwork_target(states).gather(1, actions)

		with torch.no_grad():
			labels_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

		labels = rewards + (GAMMA*labels_next*(1-dones))

		loss = criterion(predicted_targets, labels).to(device)
		self.__optimizer.zero_grad()
		loss.backward()
		self.__optimizer.step()
		self.__soft_update(self.qnetwork_local, self.qnetwork_target)


	def __soft_update(self, local_model, target_model):
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(TAU*local_param.data + (1-TAU)*target_param.data)


# Memory Replay Buffer that randomly samples experience for training Q-function
class ReplayBuffer:
	def __init__(self, batch, max_buffer):
		self.__batch_size = batch					# Batch Size of Experiences
		self.__max_buffer = max_buffer				# Max Size of Memory Buffer
		self.__memory = []							# Memory Queue 

	def add(self, action, state, reward, next_state, done):
		if len(self.__memory) > self.__max_buffer:
			self.__memory.pop(0)
		self.__memory.append([action, state, reward, next_state, done])				# Memory buffer has the following structure [[[action], [state], [reward], [next_state], [done]],
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
		torch_dones = torch.from_numpy(np.vstack(dones)).float().to(device)
		return (torch_actions, torch_states, torch_rewards, torch_next_states, torch_dones)

	def __len__(self):
		return len(self.__memory)


