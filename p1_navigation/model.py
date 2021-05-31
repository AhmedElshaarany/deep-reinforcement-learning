import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        '''
        self.FC1 = nn.Linear(state_size, 64)
        self.FC2 = nn.Linear(64, 64)
        self.FC3 = nn.Linear(64, action_size)
        '''
        self.FC1 = nn.Linear(state_size, 64)
        self.FC2 = nn.Linear(64, 128)
        self.FC3 = nn.Linear(128, 128)
        self.FC4 = nn.Linear(128, 64)
        self.FC5 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.FC1(state))
        x = F.relu(self.FC2(x))
        x = F.relu(self.FC3(x))
        x = F.relu(self.FC4(x))
        return self.FC5(x)
