import torch
import numpy as np
from matplotlib import pyplot as plt

from simulation_env import PathPlanningEnv
from reinforcement_model import Agent

def train_agent():
	num_episodes = 10
	max_t = 100000
	eps_start = 1.0
	eps_end = 0.01
	eps_decay = 0.996
	scores = []

    # Define Environment and Agent
	env = PathPlanningEnv()
	agent = Agent(env.state_size, env.action_size)

    # Start Training Loop
	eps = eps_start
	for episode in range(num_episodes):
		state = env.reset()
		score = 0.0
		for t in range(max_t):
			agent_action = agent.act(state, eps)
			print("agent_action: " + str(agent_action))
			action, state, reward, next_state, done = env.step(agent_action)
			agent.step(action, state, reward, next_state, done)
			score += reward
			if done:
				print("Agent Reached Breakpoint")
				break
			scores.append(score)
			eps = max(eps*eps_decay, eps_end)
		print('Episode: ' + str(episode + 1) + ' Score: ' + str(score))
    # Saving local QNetwork
	torch.save(agent.qnetwork_local.state_dict(), 'model/QNetwork.pth')
	env.close()
    
    # plot and hold results
	plt.plot(np.arange(len(scores)), scores)
	plt.ylabel('Score')
	plt.xlabel('Epsiode #')
	plt.show()

if __name__ == '__main__':
	train_agent()