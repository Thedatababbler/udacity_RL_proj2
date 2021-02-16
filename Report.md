### Report

[image1]:https://github.com/Thedatababbler/udacity_RL_proj2/blob/main/reacher.png

### Learning Algorithm
For this project, I used the DDPG algorithm (Deep Deterministic Policy Gradient) to solve this task. DDPG is an algorithm adapted from DQN(Deep Q-Learning)  

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


### Ideas for Future Work
Though the DDPG network was capable to solve this task, its performance on the crawler task is not good. I will try to implement the PPO algorithm soon to 
solve the crawler envrionment in the future work.

