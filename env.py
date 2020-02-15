import gym
import torch


class Env():
  def __init__(self, name = 'Pendulum-v0'):
    self._env = gym.make(name)

  def reset(self):
    state = self._env.reset()
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
  
  def step(self, action):
    state, reward, done, _ = self._env.step(action[0].detach().numpy())
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, done

  def sample_action(self):
  	return torch.tensor(self._env.action_space.sample()).unsqueeze(dim=0)