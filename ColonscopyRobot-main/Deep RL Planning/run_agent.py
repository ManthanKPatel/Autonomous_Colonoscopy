import torch

from simulation_env import PathPlanningEnv
from reinforcement_model import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_agent():
	env = PathPlanningEnv()
	agent = Agent(env.state_size, env.action_size)
	agent.qnetwork_local.load_state_dict(torch.load('model/QNetwork.pth', map_location=device))
	for episode in range(3):
		score = 0.0
		print("Starting Episode: " + str(episode))
		state = env.reset()
		for _ in range(10000):
			agent_action = agent.act(state)
			_, state, reward, next_state, done = env.step(agent_action)
			state = next_state.copy()
			score += reward
			if done:
				print("Episode Over! Maximum Achieved Score: " + str(score))
				break
	env.close()


if __name__ == '__main__':
	test_agent()