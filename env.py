import gym
import torch

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class Env():
  def __init__(self, name = 'Pendulum-v0', xml_file=None):
    self.name = name
    if xml_file is None:
      self._env = gym.make(name)
    else: 
      self._env = gym.make(name, xml_file=xml_file)
    
  def reset(self):
    state = self._env.reset()
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
  
  def step(self, action):
    state, reward, done, _ = self._env.step(action[0].detach().numpy())
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, done

  def sample_action(self):
    return torch.tensor(self._env.action_space.sample()).unsqueeze(dim=0)

  # def modify(self, xml_file):
  #   self._env = gym.make(self.name, xml_file=xml_file)





