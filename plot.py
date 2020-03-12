from collections import deque
import random
import torch
from torch import optim
from tqdm import tqdm
from hyperparams import ACTION_NOISE, OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLICY_DELAY, POLYAK_FACTOR, REPLAY_SIZE, TARGET_ACTION_NOISE, TARGET_ACTION_NOISE_CLIP, TEST_INTERVAL, UPDATE_INTERVAL, UPDATE_START
# from hyperparams import DEVICE
from hyperparams import *
from env import Env
from generalized_models import Actor, Critic, create_target_network, update_target_network
from utils import *
from gradient_penalty import gradient_penalty
import optuna
import pdb
import torch.multiprocessing as mp
import os
import pickle
import argparse
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='')
parser.add_argument('--algorithm', type=str, default='td3')
args = parser.parse_args()


env_dict = {}
env_dict['swimmer'] = ('Swimmer-v3', 'mod_envs/swimmer/', 8, 2)
env_dict['pendulum'] = ('Pendulum-v0', None, 3, 1)
env_dict['hopper'] = ('Hopper-v3', 'mod_envs/hopper/', 11, 3)
env_dict['halfcheetah'] = ('HalfCheetah-v3', 'mod_envs/halfcheetah/' , 17, 6)
env_dict['ant'] = ('Ant-v3', 'mod_envs/ant/', 111, 8)
env_dict['walker'] = ('Walker2d-v3', 'mod_envs/walker/', 17, 6)
env_dict['humanoid'] = ('Humanoid-v3', 'mod_envs/humanoid/', 376, 17)

env_name, path, state_dim, action_dim = env_dict[args.task]


def stats(rewards_list):
  time_list = [list(i) for i in itertools.zip_longest(*rewards_list)]
  means = [sum(elem)/len(elem) for elem in time_list]
  stds = [sum([(elem-mn)**2 for elem in time_slice])**(1/2)/len(time_slice) for time_slice, mn in zip(time_list, means)]
  samples = [len(time_slice) for time_slice in time_list]
  cis = [2*std*sample**(-1/2) for std, sample in zip(stds, samples)]
  # cis = [2*std for std, sample in zip(stds, samples)]
  # pdb.set_trace()
  return means, cis#stds, samples


def multiplot(steps_list, rewards_list, label_list, title):
  finished = False
  color_cycle = ['b', 'r', 'c', 'g']
  plt.figure(1)
  plt.title(title)
  means = []
  # stds = []
  # samples = []
  conf_ints = []
  for lst in rewards_list:
    m, ci = stats(lst)
    means.append(m)
    conf_ints.append(ci)

  steps_list = np.array(steps_list)
  # conf_ints = 2*stds*samples**(-1/2)
  means = np.array(means)
  conf_ints = np.array(conf_ints)
  # pdb.set_trace()
  
  plt.xlabel('Steps')
  plt.ylabel('Rewards')
  from scipy.signal import savgol_filter
  window=9
  m = savgol_filter(means, window, 2) # window size 51, polynomial order 3
  ci = savgol_filter(conf_ints, window, 2)


  for i in range(len(rewards_list)):
    color=color_cycle[i]
    # plt.plot(steps_list[i][:len(means[i])], means[i], color=color, label=label_list[i])
    # plt.fill_between(steps_list[i][:len(means[i])], np.array(means+conf_ints)[i], np.array(means-conf_ints)[i], color=color, alpha=.7)

    plt.plot(steps_list[i], m[i], color=color, label=label_list[i])
    plt.fill_between(steps_list[i], np.array(m+ci)[i], np.array(m-ci)[i], color=color, alpha=.7)

  plt.legend()
  plt.savefig(os.path.join('results', title + '.png'))
  plt.close() 


# def load_and_plot():
#   steps = [i for i in range(UPDATE_START, MAX_STEPS,TEST_INTERVAL)]
#   steps_list = [steps]*2  
#   penalty = True
#   title = 'td3_'+env_name+('_penalty' if penalty else '')
#   with open('data/' + title + '.pkl', 'rb+') as f:
#     p_train, p_test = pickle.load(f)


#   penalty = False
#   title = 'td3_'+env_name+('_penalty' if penalty else '')
#   with open('data/' + title + '.pkl', 'rb+') as f:
#     no_p_train, no_p_test = pickle.load(f)

    

#   train_rewards = [p_train, no_p_train]
#   test_rewards = [p_test, no_p_test]
#   train_rewards = np.array([p_train, no_p_train])
#   test_rewards = np.array([p_test, no_p_test])
#   labels = ['Our method', 'Baseline']

#   pdb.set_trace()
#   multiplot(steps_list, train_rewards, labels, 'td3 '+env_name+' same environment')
#   multiplot(steps_list, test_rewards, labels, 'td3 '+env_name+' modified environment')


def load_and_plot():
  steps = [i for i in range(UPDATE_START, MAX_STEPS,TEST_INTERVAL)]
  steps_list = [steps]*2  
  penalty = True
  title = 'td3_'+env_name+('_penalty' if penalty else '')
  with open('data/' + title + '.pkl', 'rb+') as f:
    p_train, p_test = pickle.load(f)


  penalty = False
  title = 'td3_'+env_name+('_penalty' if penalty else '')
  with open('data/' + title + '.pkl', 'rb+') as f:
    no_p_train, no_p_test = pickle.load(f)

    

  train_rewards = [p_train, no_p_train]
  test_rewards = [p_test, no_p_test]
  # train_rewards = np.array([p_train, no_p_train])
  # test_rewards = np.array([p_test, no_p_test])
  labels = ['Our method', 'Baseline']

  # pdb.set_trace()
  multiplot(steps_list, train_rewards, labels, args.algorithm+' '+env_name+' same environment')
  multiplot(steps_list, test_rewards, labels, args.algorithm+' '+env_name+' modified environment')


load_and_plot()
