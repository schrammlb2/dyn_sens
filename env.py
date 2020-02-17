import gym
import torch

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Walker2dEnv_Mod(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_file):
        mujoco_env.MujocoEnv.__init__(self, xml_file, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


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

  def modify(self, xml_file):
  	self._env = Walker2dEnv_Mod(xml_file)


