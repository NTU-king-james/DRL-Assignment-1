import gym
import numpy as np
from model import Actor, Critic
import torch
import torch.optim as optim

def train():
    # Create the Taxi-v3 environment
    env = gym.make('Taxi-v3', render_mode='human')
    
    # Get the observation and action spaces
    state_dim = env.observation_space.n  # Discrete observation space
    action_dim = env.action_space.n      # Discrete action space
    
    # Initialize networks
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)
    
    # Training parameters
    num_episodes = 1000
    max_steps = 100
    gamma = 0.99
    
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle the new gym API
        
        episode_reward = 0
        
        for step in range(max_steps):
            # Convert state to one-hot encoding
            state_tensor = torch.zeros(state_dim)
            state_tensor[state] = 1
            
            # Get action probabilities from actor
            action_probs = actor(state_tensor)
            
            # Sample action from probability distribution
            action = torch.multinomial(action_probs, 1).item()
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Convert next state to one-hot encoding
            next_state_tensor = torch.zeros(state_dim)
            next_state_tensor[next_state] = 1
            
            # Get value estimates
            current_value = critic(state_tensor)
            next_value = critic(next_state_tensor)
            
            # Calculate advantage
            advantage = reward + gamma * next_value.detach() - current_value.detach()
            
            # Actor loss
            log_prob = torch.log(action_probs[action])
            actor_loss = -log_prob * advantage.detach()
            
            # Critic loss
            critic_loss = advantage.pow(2)
            
            # Update networks
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        
        # Save the best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'state_dim': state_dim,
                'action_dim': action_dim,
                'best_reward': best_reward
            }, 'best_model.pth')
            print(f"New best model saved with reward: {best_reward}")
    
    env.close()

if __name__ == "__main__":
    train() 