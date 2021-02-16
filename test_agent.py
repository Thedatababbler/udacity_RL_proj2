from unityagents import UnityEnvironment
import numpy as np
from agent import Agent
import torch

def main():
    env = UnityEnvironment(file_name='Reacher20_Windows_x86_64/Reacher_Windows_x86_64/Reacher.exe')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2, num_agents=num_agents)
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor2.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic2.pth'))

    scores = 0
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations
    while True:
        #print(states.size())
        actions = agent.act(states, add_noise=False)
    
        env_info = env.step(actions)[brain_name] 
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        scores += np.mean(rewards)                         
        dones = env_info.local_done
        states = next_states
        if np.any(dones):
            break 

    print('Score: ', scores)
        
main()
    

