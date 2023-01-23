import gym
import mujoco_py
import numpy as np
from matplotlib import pyplot as plt

from reinforcement_model import Agent


# Training Parameters:
BATCH_SIZE = 256
MAX_BUFFER_SIZE = 1000000
ALPHA = 0.0003
BETA = 0.0003
GAMMA = 0.99
TAU = 0.005
EPISODES = 100                      # Number of Robot Simulations
MAX_ITERATIONS = 10000              # Maximum Iterations per episode

def train_agent():
    # define the environment for agent
    env = gym.make('FetchPickAndPlace-v1')
    # get the input dimensions of the environment and number of actions 
    num_inputs = env.observation_space.spaces["observation"].shape[0]
    num_actions = env.action_space.shape[0]
    max_action = env.action_space.high
    agent = Agent(
        BATCH_SIZE,
        MAX_BUFFER_SIZE,
        num_inputs,
        num_actions,
        max_action,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        tau=TAU
    )
    best_score = 0.0
    scores = []
    for i in range(EPISODES):
        env_dic = env.reset()
        observation = env_dic['observation']
        score = 0.0
        for t in range(MAX_ITERATIONS):
            env.render()
            action = agent.act(observation)
            next_observation_dic, reward, done, _ = env.step(action)
            next_observation = next_observation_dic['observation']
            print('reward: ' + str(reward))
            agent.step(action, observation, reward, next_observation, done)
            score += reward
            observation = np.copy(next_observation)
            if done:
                print("Robot Reach Done Condition, steps: " + str(t+1))
                agent.save_models()
                break
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        print("Episode: " + str(i+1) + " Score: " + str(score) + " Avg_score: " + str(avg_score))
    plt.plot(scores)
    plt.xlabel("Env Episodes")
    plt.ylabel("Return Score")
    plt.grid()

if __name__ == "__main__":
    train_agent()