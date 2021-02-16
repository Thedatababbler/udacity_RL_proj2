# Project2 Continuous Control

[image1]:https://github.com/Thedatababbler/udacity_RL_proj2/blob/main/reacher.png
[image2]:https://github.com/Thedatababbler/udacity_RL_proj2/blob/main/reacher.gif.gif

### Project Details
For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    Note: Since I worked on Windows environment, so there is only a windows version of agent included in this repo. Please download the respective agent to suit your system.

### Instruction
To run this project please first complete the following three steps:
1. Download the environment listed above that works on your system.

2. Clone this project to your local directory and place the file in the and same directory

3. Please refer to this page (https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up the dependencies.

Run the code:

For this project, I trained a multi-agents DDPG agent to complete the task. To train the model, please run: 

```python
python train.py
```

To watch the pre-trained smart agent to solve this task, please run: 

```python
python test_agent.py
```
Here is an example of my trained agent:

![agent][image2]







