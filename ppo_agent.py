"""
PPO Agent for Risiko.
Implements Proximal Policy Optimization using PyTorch.
Can dynamically handle valid action masking for Reinforce, Attack, and Fortify phases.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Multinomial
import pickle
import os

class PPOActorCritic(nn.Module):
    def __init__(self, input_dim=144, hidden_dims=[128, 64], output_dim=85):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        # Actor head (outputs logits for actions)
        self.actor_head = nn.Linear(hidden_dims[1], output_dim)
        # Critic head (outputs state value)
        self.critic_head = nn.Linear(hidden_dims[1], 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

class PPOAgent:
    def __init__(self, input_dim=144, rng=None, device="auto", lr=3e-4, 
                 gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, c1=0.5, c2=0.01):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = PPOActorCritic(input_dim=input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.rng = rng or np.random.default_rng()
        
        # PPO Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.c1 = c1  # value loss coef
        self.c2 = c2  # entropy coef
        self.k_epochs = 4
        
        # Memory buffer for rollout
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def set_eval(self):
        """Set to evaluation mode (no gradients)."""
        self.model.eval()

    def set_train(self):
        """Set to training mode."""
        self.model.train()

    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def store_reward(self, reward, done):
        """Usually called externally at the end of an episode."""
        if len(self.rewards) > 0:
            self.rewards[-1] = reward
            self.dones[-1] = done

    def reinforce(self, state_encoded: np.ndarray, n_armies: int, owned_territories: np.ndarray) -> np.ndarray:
        state_ts = torch.FloatTensor(state_encoded).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.model(state_ts)
            
        logits = logits.squeeze(0)[:42] # Score for the 42 territories
        mask = torch.full((42,), -1e10, device=self.device)
        mask[owned_territories] = 0.0
        
        probs = F.softmax(logits + mask, dim=-1)
        
        dist = Multinomial(n_armies, probs)
        alloc = dist.sample()
        log_prob = dist.log_prob(alloc)
        
        self.states.append(state_ts)
        self.actions.append(("reinforce", alloc, n_armies, owned_territories))
        self.log_probs.append(log_prob)
        self.values.append(value.squeeze())
        self.rewards.append(0.0)
        self.dones.append(False)
        
        return alloc.cpu().numpy().astype(np.int32)

    def attack(self, state_encoded: np.ndarray, valid_attacks: list[tuple[int, int]]) -> tuple[int, int] | None:
        if not valid_attacks:
            return None
            
        state_ts = torch.FloatTensor(state_encoded).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.model(state_ts)
            
        logits = logits.squeeze(0)
        
        # Score pairs
        scores = []
        for frm, to in valid_attacks:
            scores.append(logits[frm] + logits[42+to])
        scores.append(logits[84]) # Stop attacking action
        
        scores_ts = torch.stack(scores)
        probs = F.softmax(scores_ts, dim=-1)
        
        dist = Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        self.states.append(state_ts)
        self.actions.append(("attack", action_idx.item(), valid_attacks))
        self.log_probs.append(log_prob)
        self.values.append(value.squeeze())
        self.rewards.append(0.0)
        self.dones.append(False)
        
        idx = action_idx.item()
        if idx == len(valid_attacks):
            return None
        return valid_attacks[idx]

    def fortify(self, state_encoded: np.ndarray, valid_fortifications: list[tuple[int, int]]) -> tuple[int, int, int] | None:
        if not valid_fortifications:
            return None
            
        state_ts = torch.FloatTensor(state_encoded).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.model(state_ts)
            
        logits = logits.squeeze(0)
        
        scores = []
        for frm, to in valid_fortifications:
            scores.append(logits[frm] + logits[42+to])
        scores.append(logits[84]) # Stop fortifying
        
        scores_ts = torch.stack(scores)
        probs = F.softmax(scores_ts, dim=-1)
        
        dist = Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        self.states.append(state_ts)
        self.actions.append(("fortify", action_idx.item(), valid_fortifications))
        self.log_probs.append(log_prob)
        self.values.append(value.squeeze())
        self.rewards.append(0.0)
        self.dones.append(False)
        
        idx = action_idx.item()
        if idx == len(valid_fortifications):
            return None
            
        frm, to = valid_fortifications[idx]
        move_ratio = float(torch.sigmoid(scores_ts[idx]).item())
        return (frm, to, max(1, int(move_ratio * 50)))

    def update(self):
        """Compute PPO update based on accumulated rollout buffer."""
        if len(self.states) == 0:
            return None
            
        # 1. Compute advantages and returns
        returns = []
        advantages = []
        gae = 0
        
        values = torch.stack(self.values).cpu().numpy()
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        # Add dummy next value for GAE boundary
        next_value = 0.0 
        
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_val = next_value
                next_non_terminal = 1.0 - dones[step]
            else:
                next_val = values[step + 1]
                next_non_terminal = 1.0 - dones[step]
                
            delta = rewards[step] + self.gamma * next_val * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        old_log_probs = torch.stack(self.log_probs).detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 2. PPO Epochs
        actor_losses = []
        critic_losses = []
        entropies = []
        
        for _ in range(self.k_epochs):
            # Evaluate all step-by-step
            # Note: since action size differs dynamically, we do this in a loop rather than fully batched
            
            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            epoch_entropy = 0.0
            total_loss = 0.0
            
            for i in range(len(self.states)):
                state = self.states[i]
                action_info = self.actions[i]
                old_log_prob = old_log_probs[i]
                adv = advantages[i]
                ret = returns[i]
                
                logits, value = self.model(state)
                logits = logits.squeeze(0)
                
                if action_info[0] == "reinforce":
                    _, alloc, n_armies, owned = action_info
                    mask = torch.full((42,), -1e10, device=self.device)
                    mask[owned] = 0.0
                    probs = F.softmax(logits[:42] + mask, dim=-1)
                    dist = Multinomial(n_armies, probs)
                    new_log_prob = dist.log_prob(alloc.to(self.device))
                    entropy = torch.tensor(0.0, device=self.device) # Multinomial has no analytical entropy in PyTorch
                    
                elif action_info[0] == "attack":
                    _, action_idx, valids = action_info
                    scores = []
                    for frm, to in valids:
                        scores.append(logits[frm] + logits[42+to])
                    scores.append(logits[84])
                    scores_ts = torch.stack(scores)
                    dist = Categorical(F.softmax(scores_ts, dim=-1))
                    new_log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device))
                    entropy = dist.entropy()
                    
                elif action_info[0] == "fortify":
                    _, action_idx, valids = action_info
                    scores = []
                    for frm, to in valids:
                        scores.append(logits[frm] + logits[42+to])
                    scores.append(logits[84])
                    scores_ts = torch.stack(scores)
                    dist = Categorical(F.softmax(scores_ts, dim=-1))
                    new_log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device))
                    entropy = dist.entropy()
                
                # Ratio
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                # Surrogate Loss
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                actor_loss = -torch.min(surr1, surr2)
                
                # Value Loss
                critic_loss = F.mse_loss(value.squeeze(), ret)
                
                loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
                total_loss = total_loss + loss
                
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                epoch_entropy += entropy.item() if hasattr(entropy, 'item') else float(entropy)
                
            total_loss = total_loss / len(self.states)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            actor_losses.append(epoch_actor_loss / len(self.states))
            critic_losses.append(epoch_critic_loss / len(self.states))
            entropies.append(epoch_entropy / len(self.states))
            
        self.clear_buffer()
        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy": np.mean(entropies)
        }

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        
    def load(self, filepath):
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            print(f"Loaded PPO model from {filepath}")
        else:
            print(f"Could not load PPO model at {filepath}")
