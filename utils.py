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

def plot_with_error_bars(steps, rewards, title, label='Rewards', finished=True):
  samples = len(rewards)
  rewards = np.array(rewards)
  means = np.mean(rewards, axis=0)
  stds = np.std(rewards, axis=0)
  steps = np.array(steps)
  conf_ints = 2*stds*samples**(-1/2)

  plt.plot(steps, means, 'b-')
  plt.fill_between(np.array(steps), means+conf_ints, means-conf_ints, alpha=.7)
  plt.title(title)
  plt.xlabel('Steps')
  plt.ylabel(label)
  plt.xlim((0, MAX_STEPS))
  plt.ylim((-2000, 1000))
  if finished:
    plt.savefig(os.path.join('results', title + '.png'))
    plt.close()

def multiplot(steps_list, rewards_list, label_list, title):
  finished = False
  for i in range(len(rewards_list)):
    if i == len(rewards_list)-1:
      finished = True
    plot_with_error_bars(steps_list[i], rewards_list[i], title, label_list[i], finished=finished)

