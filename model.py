import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        return self.network(state)

# Critic network        
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        return self.network(state)

def get_action(obs, actor):
    # Convert observation to one-hot encoding
    state_dim = actor.network[0].in_features
    state_tensor = torch.zeros(state_dim)
    state_tensor[obs] = 1
    
    # Get action probabilities from actor
    action_probs = actor(state_tensor)
    
    # Sample action from probability distribution
    action = torch.multinomial(action_probs, 1).item()
    
    return action