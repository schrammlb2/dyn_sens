import os
import matplotlib.pyplot as plt
from hyperparams import MAX_STEPS


steps, rewards = [], []

def mean(lst):
  return sum(lst)/len(lst)

def std(lst):
  mn = mean(lst)
  return [(elem - mn)**2 for elem in lst]**.5

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

def plot_with_error_bars(steps, rewards, title):
  samples = len(rewards)
  means = []
  stds = []
  train_std = []
  for i in range(samples):
    means.append(mean(rewards[i]))
    stds.append(std(rewards[i])*samples**(-.5)) # confidence intervals

  means = np.array(means)
  stds = np.array(stds)

  plt.plot(steps, rewards, 'b-')
  plt.fill_between(np.array(steps), means+stds, means-stds, alpha=.7)
  plt.title(title)
  plt.xlabel('Steps')
  plt.ylabel('Rewards')
  plt.xlim((0, MAX_STEPS))
  plt.ylim((-2000, 1000))
  plt.savefig(os.path.join('results', title + '.png'))
