import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchviz
from tqdm import tqdm
import tempfile



import os
import random

class ReplayBuffer:
    def __init__(self):
        self.episodes = []

    def append(self, episode):
        self.episodes.append(episode)
        
    def sample_batch(self, batch_size):
        # First, flatten all episodes into a list of steps
        all_steps = []
        for episode in self.episodes:
            all_steps.extend(episode.steps)
        
        # Randomly sample transitions from all_steps
        batch_steps = random.sample(all_steps, batch_size)
        
        # Build the batch dictionary
        batch = {
            "states": torch.stack([torch.tensor(step.state, dtype=torch.float32) for step in batch_steps]),
            "actions": torch.stack([torch.tensor(step.action, dtype=torch.float32) for step in batch_steps]),
            "rewards": torch.stack([torch.tensor(step.reward, dtype=torch.float32) for step in batch_steps]),
            "next_states": torch.stack([torch.tensor(step.next_state, dtype=torch.float32) for step in batch_steps])
        }
        return batch

class Episode:
    def __init__(self):
        self.steps = []
    
    def append(self, state, action, reward, next_state):
        step = Step(state, action, reward, next_state)
        self.steps.append(step)
    
class Step:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class QNetwork(nn.Module):
    def __init__(self, state_action_dim=6, q_value_dim=1, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.q_value_layer = nn.Linear(hidden_dim, q_value_dim)
        
    def forward(self, state_action):
        x = F.relu(self.fc1(state_action))
        
        x = F.relu(self.fc2(x))
        
        q_value = self.q_value_layer(x)
        
        return q_value
        
    def qvalue(self, state_action):
        q_value = self.forward(state_action)
        return q_value

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))

        x = F.relu(self.fc2(x))

        mean = self.mean_layer(x)

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        
        print("Mean:", mean, "\nStd:", std, "\nAction:", action)
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        squashed_action = torch.tanh(action)
        return squashed_action, log_prob




#       Control Scripts       #


trained_policy = None

def train(sim):
    
    policy = PolicyNetwork()
    qnetwork = QNetwork()
    rb = ReplayBuffer()
    
    policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    q_optimizer = optim.Adam(qnetwork.parameters(), lr=3e-4)
    
    state = sim.observe()

    num_iterations, num_action_episodes, len_episode = 100, 10, 20
    gamma, alpha = 0.99, 0.2
    for iteration in tqdm(range(0, num_iterations), position=0):
        for action_episode in tqdm(range(0, num_action_episodes), position=1, leave=False):
            e = Episode()
            for episode in range(0, len_episode):
                action = policy.sample(state)[0].detach().numpy()
                sim.act(action)
                reward = sim.reward()
                next_state = sim.observe()
                e.append(state, action, reward, next_state)
                state = next_state
            sim.reset()
            rb.append(e)
        
        gradient_steps = 10
        for gradient_step in tqdm(range(0, gradient_steps), position=1, leave=False):
            batch = rb.sample_batch(2)
            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            next_states = batch['next_states']
            # Critic update #
            state_actions = torch.cat((states, actions), dim=-1)
            q_current = qnetwork(state_actions)
            
            with torch.no_grad():
                next_actions, next_log_probs = policy.sample(next_states)
                next_state_actions = torch.cat((next_states, next_actions), dim=-1)
                q_next = qnetwork(next_state_actions)
                target_q = rewards + gamma * (q_next - alpha * next_log_probs)
                
            q_loss = F.mse_loss(q_current, target_q)
            
            # Actor update #
            new_actions, log_probs = policy.sample(states)  
            new_state_actions = torch.cat((states, new_actions), dim=-1)
            q_val_new = qnetwork(new_state_actions)
            policy_loss = (alpha * log_probs - q_val_new).mean()
            
            # Backpropogation #
            policy_optimizer.zero_grad()
            q_optimizer.zero_grad()    
            
            policy_loss.backward(retain_graph=True)  # retain_graph to use the graph further
            q_loss.backward()  
            
            policy_optimizer.step()
            q_optimizer.step()  
            
    global trained_policy
    trained_policy = policy
    safe_save_model(trained_policy, "trained_policy.pt", save_state_dict=True)
    
def test(sim):
    import time
    sim.has_renderer = True
    sim.reset()
    global trained_policy
    state = sim.observe()
    num_steps = 100
    for step in range(0, num_steps):
        print(state)
        action = trained_policy.sample(state)[0].detach().numpy()
        print(action)        
        sim.act(action) 
        state = sim.observe()
        print(sim.reward)
        time.sleep(1)
              
       
def test_single_SAR(sim):
    policy = PolicyNetwork()

    state = sim.observe()
    
    # create tensor out of human-readable "state"
    state = torch.tensor(state)
    
    action, log_prob = policy.sample(state)

def load_saved_model(model_path):
    global trained_policy
    trained_policy = PolicyNetwork()
    trained_policy.load_state_dict(torch.load(model_path))
    trained_policy.eval()  

def safe_save_model(model, filename, save_state_dict=True):
    """
    Safely save a PyTorch model or its state_dict to a file using an atomic write.
    
    Parameters:
        model (torch.nn.Module): The model to save.
        filename (str): The target filename where the model will be saved.
        save_state_dict (bool): If True, only the model's state_dict will be saved.
                                Otherwise, the entire model is saved.
    """
    # Choose the data to save
    data_to_save = model.state_dict() if save_state_dict else model

    # Get the target directory from filename
    target_dir = os.path.dirname(os.path.abspath(filename))
    
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Use a temporary file in the same directory for atomic write.
    with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as tmp_file:
        temp_filename = tmp_file.name
        torch.save(data_to_save, tmp_file)
    
    # Atomically replace the target file with the temporary file.
    os.replace(temp_filename, filename)
    print(f"Model successfully saved to {filename}")    
    

  

