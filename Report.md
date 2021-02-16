### Report

[image1]:https://github.com/Thedatababbler/udacity_RL_proj2/blob/main/rewards.png
[image2]:https://github.com/Thedatababbler/udacity_RL_proj2/blob/main/rewards3.png

### Learning Algorithm
For this project, I used the DDPG algorithm (Deep Deterministic Policy Gradient) to solve this task. 
DDPG is an algorithm adapted from DQN(Deep Q-Learning) which can deal with continuous action space, while the origin DQN can only deal with discrete action space.

Both DDPG and DQN are deterministic algorithm which means they are trying to fitting a function that can determin what action to take given states as input. 
However, unlike DQN, DDPG also adopt the actor-critic idea to learn the deterministic function. 

An actor-critic algorithm need to train two seperate networks -- the actor network and the critic network. The actor network is reponsible for determing action given states input, while the critic network tries to tell the actor network whether such action is good or bad in terms of reward gaining. For example, the critic network in DDPG is the
Q network that expect the future reward without extend the whole episode for a given action. On the other hand, the actor network in the DDPG only cares about the value of the
Q network rather than the future reward of an action.

In this project, for both the actor network and the critic network, I used a two-layer linear network to fit the functions. The loss for the critic network is the mean square error between the local network Q-value and the target network. And the loss for the actor network is simply the negative Q-value of the local critic network since we need to
optimize such value. 

I also use the OUnoise in the agent for better training performance. The soft update rate was set as 1e-3, and the learning rate is 1e-4 for both network.

#### Single Agent DDPG
I started with training the DDPG network with a single agent. I found that the network has a difficulty to converge and fluctuate around with a reward value near +30, but it
was never stable enough to give me a larger than 30 reward value over consecutive 100 episodes. 

#### Distributed DDPG
After trained a DDPG with the single version environment, I also explored the multi-agent version of DDPG. The multi-agent version achieve a 30ish reward faster comparing to
the single agent version, however, the reward value started to drop after some episodes due to the overoptimistic evaluation problem. 

After I halved the Tau value to let the target network update less frequently and applied the following trick to my code, the reward became much more stable.

```python
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
    self.critic_optimizer.step()
      )
```

### Plot of Rewards
After about 100 episodes, the reward of the multi-agent DDPG network became stable and over 30

![reward][image1]

![reward2][image2]


### Ideas for Future Work
Though the DDPG network was capable to solve this task, its performance on the crawler task is not good. I will try to implement the PPO algorithm soon to 
solve the crawler envrionment in the future work.

