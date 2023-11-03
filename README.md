# Autonomous_Colonoscopy

### Classification ###

### Deep RL Grasping ###

The open ai gym environment used by RL Agent
```
env = gym.make('FetchPickAndPlace-v1')
```
![grasping_robot](/ColonscopyRobot-main/docs/grasping/axes_of_robot.png)

The environment observation consists of the following:
```obs = np.concatenate([ grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(), object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel, ])```

![observations_of_robot](/ColonscopyRobot-main/docs/grasping/observations_of_grasping_robot.png)

The input action in the FetchPickAndPlace environments is expected as an posistional increments. The first three values define an increment of the current position tool center point of the robot. The last value defines the spacing between the robots grippers.

![actions_of_robot](/ColonscopyRobot-main/docs/grasping/action_space_grasping.png)

The reward function used in the environment.
```
def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.
    d = goal_distance(achieved_goal, goal)
    if self.reward_type == 'sparse':
        return -(d > self.distance_threshold).astype(np.float32)
    else:
        return -d
```

The evaluation metric used in reward between the gripper and target object.
```
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
```

The Soft-Actor-Critic Algorithm

![soft-actor-critic-alogrithm](/ColonscopyRobot-main/docs/grasping/soft-actor-critic-algorithm.jpg)


#### Training the Q-Function or Actor 1 and Actor 2 ####

The soft Q-function parameters can be trainined to minimize the soft Bellman residual

![equation_7_and_8](/ColonscopyRobot-main/docs/grasping/equation1.jpg)

which again can be optimized with stochastic gradients.

![equation_9](/ColonscopyRobot-main/docs/grasping/equation2.jpg)

#### Training the Value-Functions ####

The state value function approximates the soft value. There is no need in principle to include a separate function approximator for the state value. But including separate function approximator ``` target_value``` for the soft value can stabilize training and is convenient to train simulataneously with the other networks.

The soft value function is trained to minimize the squared residual error.

![equation_5](/ColonscopyRobot-main/docs/grasping/equation3.jpg)

The gradient of equation 5 can be estimated with an unbiased estimator.

![equation_6](/ColonscopyRobot-main/docs/grasping/equation4.jpg)

#### Training the Policy Function ####

The policy parameters can be learned by directly minimizing the expected KL-divergence.

![equation_10](/ColonscopyRobot-main/docs/grasping/equation5.jpg)

To avoid backpropagating errors a reparameterization tick is used to make sure that sampling from policy is differentiable. The policy is now parameterized as follows:

![eqaution_11](/ColonscopyRobot-main/docs/grasping/equation6.jpg)

where elispon is input noise vector, sampled from fixed gaussian distribution. Rewriting the objective Bellman function.

![equation_12](/ColonscopyRobot-main/docs/grasping/equation7.jpg)

We can approximate the gradient as the following.

![equation_13]('/ColonscopyRobot-main/docs/grasping/equation8.jpg')

Block Diagram putting training all together.


References:
- https://upcommons.upc.edu/bitstream/handle/2117/185242/thesis.pdf?sequence=1&isAllowed=y
- https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665
- https://arxiv.org/abs/1801.01290
- https://arxiv.org/abs/1812.05905
- https://github.com/openai/mujoco-py
- https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC
- https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
- https://github.com/ku2482/soft-actor-critic.pytorch/blob/master/code/agent.py
- https://gregorygundersen.com/blog/2018/04/29/reparameterization/


### Deep RL Planning ###

Colon Environment 

![DQN_Flowchart]('/ColonscopyRobot-main/docs/planning/program_flow_chart.png)

References:
- https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/


### Segmentation ###

### SLAM ###
