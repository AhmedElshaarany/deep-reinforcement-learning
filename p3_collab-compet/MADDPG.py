import random
import torch
from ddpg_agent import Agent
from utils import ReplayBuffer, OUNoise

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
LEARN_FREQUENCY = 3
N_LEARNING_EXP = 4

class MADDPG():
    
    """Creates set of agents to interact with and learn from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize set of agents object.
         Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        
        self.agents = []
        for _ in range(num_agents):
            self.agents.append(Agent(state_size=state_size, action_size=action_size, random_seed=10))
            
        # define shared replay buffer
        self.shared_memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # create counter for learning update
        self.step_count = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience / reward for each agent
        for i in range(self.num_agents):
            self.shared_memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        self.step_count += 1

        # Learn, if enough samples are available in memory
        if len(self.shared_memory) > BATCH_SIZE:
            if(self.step_count % LEARN_FREQUENCY == 0):
                # learn N_LEARNING_EXP for each agent
                for _ in range(N_LEARNING_EXP):
                    for i in range(self.num_agents):
                        experiences = self.shared_memory.sample()
                        self.agents[i].learn(experiences, GAMMA)
                        
    def act(self, states, add_noise=True):
        agents_actions = []
        
        #append actions for each agent
        for i in range(self.num_agents):
            agents_actions.append(self.agents[i].act(states[i]))
            
        return agents_actions
    
    def reset(self):
        # reset every agent
        for i in range(self.num_agents):
            self.agents[i].reset()