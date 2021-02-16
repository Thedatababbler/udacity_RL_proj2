from unityagents import UnityEnvironment
import numpy as np
from collections import namedtuple, deque
from model import Actor, Critic
from agent import Agent
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    env = UnityEnvironment(file_name='Reacher20_Windows_x86_64/Reacher_Windows_x86_64/Reacher.exe')

    # get the default brain
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
    def ddpg(n_episodes=10, max_t=1000, print_every=100):
        scores_deque = deque(maxlen=print_every)
        scores = []
        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            agent.reset()
            score = np.zeros(num_agents) 
            while True:
                actions = agent.act(states)
                env_info = env.step(actions)[brain_name] 
                next_states = env_info.vector_observations
                rewards = env_info.rewards                         
                dones = env_info.local_done
                agent.step(states, actions, rewards, next_states, dones)
                score += env_info.rewards
                states = next_states
                if np.any(dones):
                    break 
            scores.append(np.mean(score))
            scores_deque.append(np.mean(score))
            print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores), np.mean(scores_deque)), end="")
            if np.mean(scores_deque) >= 30.0 and i_episode >= 100:
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                break 
            if i_episode % print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                
        return scores

    scores = ddpg()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

if __name__ == "__main__":
    train()
