import gym
import torch

import numpy as np
from gym import utils
from hyperparams import *
from modified_envs import *


class Env():
  def __init__(self, name = 'Pendulum-v0', xml_file=None):
    self.name = name
    if xml_file is None:
      self._env = gym.make(name)
    else: 
      try:
        self._env = gym.make(name, xml_file=xml_file)
      except:
        if 'InvertedPendulum' in name:
          self._env=InvertedPendulumEnv_Mod(xml_file)
        else:
          #Other environments go here
          pass
    
  def reset(self):
    state = self._env.reset()
    return torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(dim=0)
  
  def step(self, action):
    state, reward, done, _ = self._env.step(action[0].detach().cpu().numpy())
    return torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(dim=0), reward, done

  def sample_action(self):
    return torch.tensor(self._env.action_space.sample(), device=DEVICE).unsqueeze(dim=0)

  # def modify(self, xml_file):
  #   self._env = gym.make(self.name, xml_file=xml_file)





