import gym
import mujoco_py
import numpy as np

from reinforcement_model import Agent

def test_agent():
    # define the environment for agent
    env = gym.make('FetchPickAndPlace-v1')
    
    # get the input dimensions of the environment and number of actions 
    num_inputs = env.observation_space.spaces["observation"].shape[0]
    num_actions = env.action_space.shape[0]
    num_goals = env.observation_space.spaces["desired_goal"].shape[0]
    max_action = env.action_space.high
    reward_range = env.reward_range
    print("Grasping Robot Environment Info:")
    print("State Shape: " + str(num_inputs))
    print("Actions:  " + str(num_actions))
    print("Action Bounds: " + str(max_action))
    print("Num Goals: " + str(num_goals))
    print("Reward Range: " + str(reward_range))

    for i_episode in range(20):
        observation = env.reset()
        for t in range(1000):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

if __name__ == "__main__":
    test_agent()