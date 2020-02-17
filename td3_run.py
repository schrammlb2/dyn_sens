from collections import deque
import random
import torch
from torch import optim
from tqdm import tqdm
from hyperparams import ACTION_NOISE, OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLICY_DELAY, POLYAK_FACTOR, REPLAY_SIZE, TARGET_ACTION_NOISE, TARGET_ACTION_NOISE_CLIP, TEST_INTERVAL, UPDATE_INTERVAL, UPDATE_START
from env import Env, Walker2dEnv_Mod
# from models import Actor, Critic, create_target_network, update_target_network
from generalized_models import Actor, Critic, create_target_network, update_target_network
from utils import plot
from gradient_penalty import gradient_penalty
import optuna

env_name = 'Walker2d-v2'
path = 'mod_envs/walker/'
# env_name = 'HalfCheetah-v2'
state_dim = 17
action_dim=6


MAX_STEPS = 15000


def test(actor):
  with torch.no_grad():
    env = Env(env_name)
    state, done, total_reward = env.reset(), False, 0
    # while not done:
    for i in range(1000):
      action = torch.clamp(actor(state), min=-1, max=1)  # Use purely exploitative policy at test time
      state, reward, done = env.step(action)
      total_reward += reward
      if done: 
        break
    return total_reward

def test_transfer(actor, path):
  import os
  for xml in os.listdir(path):
    filename = os.getcwd() + '/'+ path + xml
    with torch.no_grad():
      env = Env()
      env.modify(filename)
      state, done, total_reward = env.reset(), False, 0
      for i in range(1000):
        action = torch.clamp(actor(state), min=-1, max=1)  # Use purely exploitative policy at test time
        state, reward, done = env.step(action)
        total_reward += reward
        if done: 
          break
      return total_reward


def train_td3(penalty = False, epsilon=.03, actor_LEARNING_RATE=LEARNING_RATE, critic_LEARNING_RATE=LEARNING_RATE):
  env = Env(env_name)

  actor = Actor(HIDDEN_SIZE, stochastic=False, layer_norm=True, state_dim=state_dim, action_dim=action_dim)
  critic_1 = Critic(HIDDEN_SIZE, state_action=True, layer_norm=True, state_dim=state_dim, action_dim=action_dim)
  critic_2 = Critic(HIDDEN_SIZE, state_action=True, layer_norm=True, state_dim=state_dim, action_dim=action_dim)
  target_actor = create_target_network(actor)
  target_critic_1 = create_target_network(critic_1)
  target_critic_2 = create_target_network(critic_2)
  # actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
  # critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
  actor_optimiser = optim.Adam(actor.parameters(), lr=actor_LEARNING_RATE)
  critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=critic_LEARNING_RATE)
  D = deque(maxlen=REPLAY_SIZE)

  steps = []
  rewards = []
  transfer_rewards = []

  state, done = env.reset(), False
  pbar = tqdm(range(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)
  total_reward = 0
  total_transfer_reward = 0
  for step in pbar:
    with torch.no_grad():
      if step < UPDATE_START:
        # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
        # action = torch.tensor([[2 * random.random() - 1]])
        action = env.sample_action()
      else:
        # Observe state s and select action a = clip(μ(s) + ε, a_low, a_high)
        action = torch.clamp(actor(state) + ACTION_NOISE * torch.randn(1, 1), min=-1, max=1)
      # Execute a in the environment and observe next state s', reward r, and done signal d to indicate whether s' is terminal
      next_state, reward, done = env.step(action)
      # Store (s, a, r, s', d) in replay buffer D
      D.append({'state': state, 'action': action, 'reward': torch.tensor([reward]), 'next_state': next_state, 'done': torch.tensor([done], dtype=torch.float32)})
      state = next_state
      # If s' is terminal, reset environment state
      if done:
        state = env.reset()

    if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
      # pbar.set_description('Step: %i | Reward: %f | Transfer Reward: %f' % (step, total_reward, total_transfer_reward))
      # Randomly sample a batch of transitions B = {(s, a, r, s', d)} from D
      batch = random.sample(D, BATCH_SIZE)
      batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}

      # Compute target actions with clipped noise (target policy smoothing)
      target_action = torch.clamp(target_actor(batch['next_state']) + torch.clamp(TARGET_ACTION_NOISE * torch.randn(1, 1), min=-TARGET_ACTION_NOISE_CLIP, max=TARGET_ACTION_NOISE_CLIP), min=-1, max=1)
      # Compute targets (clipped double Q-learning)
      if penalty: 
        tc1 = gradient_penalty(target_critic_1, batch['next_state'], target_action, epsilon=epsilon)
        tc2 = gradient_penalty(target_critic_2, batch['next_state'], target_action, epsilon=epsilon)     
      else:
        tc1 = target_critic_1(batch['next_state'], target_action)
        tc2 = target_critic_2(batch['next_state'], target_action)
      y = batch['reward'] + DISCOUNT * (1 - batch['done']) * torch.min(tc1, tc2)

      # Update Q-functions by one step of gradient descent
      value_loss = (critic_1(batch['state'], batch['action']) - y).pow(2).mean() + (critic_2(batch['state'], batch['action']) - y).pow(2).mean()
      critics_optimiser.zero_grad()
      value_loss.backward()
      critics_optimiser.step()

      if step % (POLICY_DELAY * UPDATE_INTERVAL) == 0:
        # Update policy by one step of gradient ascent
        policy_loss = -critic_1(batch['state'], actor(batch['state'])).mean()
        actor_optimiser.zero_grad()
        policy_loss.backward()
        actor_optimiser.step()

        # Update target networks
        update_target_network(critic_1, target_critic_1, POLYAK_FACTOR)
        update_target_network(critic_2, target_critic_2, POLYAK_FACTOR)
        update_target_network(actor, target_actor, POLYAK_FACTOR)

    if step > UPDATE_START and step % TEST_INTERVAL == 0:
      actor.eval()
      total_reward = test(actor)
      total_transfer_reward = test_transfer(actor, path)
      pbar.set_description('Step: %i | Reward: %f | Transfer Reward: %f' % (step, total_reward, total_transfer_reward))
      steps.append(step)
      rewards.append(total_reward)
      transfer_rewards.append(total_transfer_reward)
      # plot_static(steps, rewards, 'td3_'+env_name+('_penalty' if penalty else ''))
      # plot_static(step, transfer_rewards, 'td3_'+env_name+('_penalty' if penalty else ''))
      actor.train()
  return rewards, transfer_rewards


def objective(trial):
  epsilon = trial.suggest_loguniform('epsilon', .005, .05)
  actor_LEARNING_RATE = trial.suggest_loguniform('epsilon', LEARNING_RATE*.3, LEARNING_RATE*3)
  critic_LEARNING_RATE = LEARNING_RATE#trial.suggest_loguniform('epsilon', LEARNING_RATE*.3, LEARNING_RATE*3)

  return -sum(train_td3(penalty=True, epsilon=epsilon)[0])#, actor_LEARNING_RATE=actor_LEARNING_RATE, critic_LEARNING_RATE=critic_LEARNING_RATE)

def hyperparameter_search():
  study = optuna.create_study()
  study.optimize(objective, n_trials=100)
  print(study.best_params)


# def train_parallel(*args):
#   return train_td3(*args)

def evaluate():
  samples = 10
  train_rewards = []
  test_rewards = []

  steps = [i for i in range(UPDATE_START, MAX_STEPS ,UPDATE_INTERVAL)]

  for i in range(samples):
    rewards, transfer_rewards= train_td3(penalty=True, epsilon=.05)
    train_rewards.append(rewards)
    test_rewards.append(transfer_rewards)

  title = 'td3_'+env_name+('_penalty' if penalty else '')
  plot_with_error_bars(steps, train_rewards, title)

  title = 'td3_transfer_'+env_name+('_penalty' if penalty else '')
  plot_with_error_bars(steps, test_rewards, title)



evaluate()
