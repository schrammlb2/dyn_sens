import os
import matplotlib.pyplot as plt
from hyperparams import MAX_STEPS
import pdb
import numpy as np


steps, rewards = [], []

def mean(lst):
  return sum(lst)/len(lst)

def std(lst):
  mn = mean(lst)
  return sum([(elem - mn)**2 for elem in lst])**.5

def plot(step, reward, title):
  steps.append(step)
  rewards.append(reward)
  plt.plot(steps, rewards, 'b-')
  plt.title(title)
  plt.xlabel('Steps')
  plt.ylabel('Rewards')
  plt.xlim((0, MAX_STEPS))
  plt.ylim((-2000, 1000))
  plt.savefig(os.path.join('results', title + '.png'))


def multiplot(steps_list, rewards_list, label_list, title):
  finished = False
  color_cycle = ['b', 'r', 'c', 'g']
  plt.figure(1)
  plt.title(title)
  # pdb.set_trace()
  rewards_list = np.array(rewards_list)
  samples = rewards_list.shape[1]
  means = np.mean(rewards_list, axis=1)
  stds = np.std(rewards_list, axis=1)
  steps_list = np.array(steps_list)
  conf_ints = 2*stds*samples**(-1/2)

  
  plt.xlabel('Steps')
  plt.ylabel('Rewards')

  for i in range(len(rewards_list)):
    color=color_cycle[i]
    plt.plot(steps_list[i], means[i], color=color, label=label_list[i])
    plt.fill_between(steps_list[i], np.array(means+conf_ints)[i], np.array(means-conf_ints)[i], color=color, alpha=.7)
    # plot_with_error_bars(steps_list[i], rewards_list[i], title, label_list[i], finished=finished, color=color_cycle[i])
  plt.legend()
  plt.savefig(os.path.join('results', title + '.png'))
  plt.close() 