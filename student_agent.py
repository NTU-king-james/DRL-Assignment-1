# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
from model import Actor

# Load the trained model
try:
    checkpoint = torch.load('best_model.pth')
    actor = Actor(checkpoint['state_dim'], checkpoint['action_dim'])
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()  # Set to evaluation mode
    print("Successfully loaded trained model")
except Exception as e:
    print(f"Error loading model: {e}")
    actor = None

def get_action(obs):
    if actor is None:
        # Fallback to random action if model is not loaded
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # Convert observation to one-hot encoding
    state_dim = actor.network[0].in_features
    state_tensor = torch.zeros(state_dim)
    state_tensor[obs] = 1
    
    # Get action probabilities from actor
    with torch.no_grad():
        action_probs = actor(state_tensor)
    
    # Sample action from probability distribution
    action = torch.multinomial(action_probs, 1).item()
    
    return action


