from collections import deque
import random
import torch
from torch import optim
from tqdm import tqdm
from hyperparams import ACTION_NOISE, OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLICY_DELAY, POLYAK_FACTOR, REPLAY_SIZE, TARGET_ACTION_NOISE, TARGET_ACTION_NOISE_CLIP, TEST_INTERVAL, UPDATE_INTERVAL, UPDATE_START
# from hyperparams import DEVICE
from hyperparams import *
from env import Env#, Walker2dEnv_Mod
# from models import Actor, Critic, create_target_network, update_target_network
from generalized_models import Actor, SoftActor, Critic, create_target_network, update_target_network
from utils import *
from gradient_penalty import gradient_penalty
import optuna
import pdb
import torch.multiprocessing as mp
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='')
# parser.add_argument('--penalty', type=bool, default=True)
parser.add_argument('--no_penalty', default=True, action='store_false', help='Bool type')
parser.add_argument('--algorithm', type=str, default='sac')
args = parser.parse_args()

# MAX_STEPS = 15000
# try: 
#   mp.set_start_method("spawn")
# except:
#   pass
def test(actor):
  with torch.no_grad():
    env = Env(env_name)
    state, done, total_reward = env.reset(), False, 0
    # while not done:
    for i in range(1000):
      a=actor(state)
      if type(a) != torch.Tensor:
        a = a.mean
      action = torch.clamp(a, min=-1, max=1)  # Use purely exploitative policy at test time
      state, reward, done = env.step(action)
      total_reward += reward
      if done: 
        break
    return total_reward


def test_worst_case(actor, critic_1, critic_2):
  env = Env(env_name)
  state, done, total_reward = env.reset(), False, 0
  epsilon = .03
  for i in range(1000):
    a=actor(state)
    if type(a) != torch.Tensor:
      a = a.mean
      # Use purely exploitative policy at test time
    g1 = action_gradient(critic_1, state, action, epsilon=.1)
    g2 = action_gradient(critic_2, state, action, epsilon=.1)
    bad_action = action-(g1+g2)/2*epsilon
    action = torch.clamp(bad_action, min=-1, max=1)
    state, reward, done = env.step(action)
    total_reward += reward
    if done: 
      break
  return total_reward


def test_action_noise(actor):
  for i in range(4):
    sigma = .03
    with torch.no_grad():
      env = Env(env_name)
      state, done, total_reward = env.reset(), False, 0
      # while not done:
      for i in range(1000):
        s = np.random.normal(0, sigma, action_dim)
        a=actor(state) + s
        if type(a) != torch.Tensor:
          a = a.mean
        action = torch.clamp(a, min=-1, max=1)  # Use purely exploitative policy at test time
        state, reward, done = env.step(action)
        total_reward += reward
        if done: 
          break
      return total_reward

def test_transfer(actor, path):
  if path is not None:
    filenames = [os.getcwd() + '/'+ path + xml for xml in os.listdir(path) if xml.endswith('.xml') ]
    # for xml in os.listdir(path):
    fn = sorted(filenames, key = lambda x: len(x))[1:]

    randomize_xml(fn[0], scale=0.3, count=4)
    # for filename in filenames:
    reward_list = []
    for filename in fn:
      with torch.no_grad():
        env = Env(env_name, xml_file=filename)
        # env.modify(filename)
        state, done, total_reward = env.reset(), False, 0
        for i in range(1000):
          a=actor(state)
          if type(a) != torch.Tensor:
            a = a.mean
          action = torch.clamp(a, min=-1, max=1)  # Use purely exploitative policy at test time
          state, reward, done = env.step(action)
          total_reward += reward
          if done: 
            break
        reward_list.append(total_reward)

    return mean(reward_list)
  else:
    return 0


def train_td3(penalty = False, epsilon=.02, actor_LEARNING_RATE=LEARNING_RATE, critic_LEARNING_RATE=LEARNING_RATE, h = HIDDEN_SIZE):
  env = Env(env_name)

  actor = Actor(h, stochastic=False, layer_norm=True, state_dim=state_dim, action_dim=action_dim).to(DEVICE)
  critic_1 = Critic(h,  state_action=True, layer_norm=True, state_dim=state_dim, action_dim=action_dim).to(DEVICE)
  critic_2 = Critic(h,  state_action=True, layer_norm=True, state_dim=state_dim, action_dim=action_dim).to(DEVICE)
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
        action = env.sample_action()
      else:
        # pdb.set_trace()
        # Observe state s and select action a = clip(μ(s) + ε, a_low, a_high)
        action = torch.clamp(actor(state) + ACTION_NOISE * torch.randn(1, 1, device=DEVICE), min=-1, max=1)
      # Execute a in the environment and observe next state s', reward r, and done signal d to indicate whether s' is terminal
      next_state, reward, done = env.step(action)
      # Store (s, a, r, s', d) in replay buffer D
      D.append({'state': state, 'action': action, 'reward': torch.tensor([reward], device=DEVICE), 'next_state': next_state, 'done': torch.tensor([done], dtype=torch.float32, device=DEVICE)})
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
      target_action = torch.clamp(target_actor(batch['next_state']) + torch.clamp(TARGET_ACTION_NOISE * torch.randn(1, 1, device=DEVICE), min=-TARGET_ACTION_NOISE_CLIP, max=TARGET_ACTION_NOISE_CLIP), min=-1, max=1)
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
      # if not MULTIPROCESSING:
      pbar.set_description('Step: %i | Reward: %f | Transfer Reward: %f' % (step, total_reward, total_transfer_reward))
      steps.append(step)
      rewards.append(total_reward)
      transfer_rewards.append(total_transfer_reward)
      # plot_static(steps, rewards, 'td3_'+env_name+('_penalty' if penalty else ''))
      # plot_static(step, transfer_rewards, 'td3_'+env_name+('_penalty' if penalty else ''))
      actor.train()
  # pdb.set_trace()
  return rewards, transfer_rewards#, steps


def train_sac(penalty = True, epsilon=.03):
  env = Env(env_name)
  actor = SoftActor(HIDDEN_SIZE, state_dim=state_dim, action_dim=action_dim).to(DEVICE)
  critic_1 = Critic(HIDDEN_SIZE, state_action=True, state_dim=state_dim, action_dim=action_dim).to(DEVICE)
  critic_2 = Critic(HIDDEN_SIZE, state_action=True, state_dim=state_dim, action_dim=action_dim).to(DEVICE)
  value_critic = Critic(HIDDEN_SIZE, state_dim=state_dim, action_dim=action_dim).to(DEVICE)
  target_value_critic = create_target_network(value_critic).to(DEVICE)
  actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
  critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
  value_critic_optimiser = optim.Adam(value_critic.parameters(), lr=LEARNING_RATE)
  D = deque(maxlen=REPLAY_SIZE)

  steps = []
  rewards = []
  transfer_rewards = []
  # worst_case_rewards = []

  # test_functions = 
  # reward_categories = []

  state, done = env.reset(), False
  pbar = tqdm(range(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)
  for step in pbar:
    with torch.no_grad():
      if step < UPDATE_START:
        # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
        action = env.sample_action()
      else:
        # Observe state s and select action a ~ μ(a|s)
        action = actor(state).sample()
      # Execute a in the environment and observe next state s', reward r, and done signal d to indicate whether s' is terminal
      if (action != action).all():
        pdb.set_trace()
      next_state, reward, done = env.step(action)
      # Store (s, a, r, s', d) in replay buffer D
      D.append({'state': state, 'action': action, 'reward': torch.tensor([reward], device=DEVICE), 'next_state': next_state, 'done': torch.tensor([done], dtype=torch.float32, device=DEVICE)})
      state = next_state
      # If s' is terminal, reset environment state
      if done:
        state = env.reset()

    if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
      # Randomly sample a batch of transitions B = {(s, a, r, s', d)} from D
      batch = random.sample(D, BATCH_SIZE)
      batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}

      # Compute targets for Q and V functions
      if penalty:
        tvc = gradient_penalty(target_value_critic, batch['next_state'])
      else:
        tvc = target_value_critic(batch['next_state'])

      y_q = batch['reward'] + DISCOUNT * (1 - batch['done']) * tvc
      policy = actor(batch['state'])
      action = policy.rsample()  # a(s) is a sample from μ(·|s) which is differentiable wrt θ via the reparameterisation trick
      log_probs = policy.log_prob(action)
      NaNMask = (log_probs!=log_probs)
      log_probs[NaNMask] = 0
      weighted_sample_entropy = ENTROPY_WEIGHT * log_probs.sum(dim=1)  # Note: in practice it is more numerically stable to calculate the log probability when sampling an action to avoid inverting tanh
      if penalty:
        c1 = gradient_penalty(critic_1, batch['state'], action.detach())
        c2 = gradient_penalty(critic_2, batch['state'], action.detach())
      else:
        c1 = critic_1(batch['state'], action.detach())
        c2 = critic_2(batch['state'], action.detach())

      y_v = torch.min(c1, c2) - weighted_sample_entropy.detach()

      # Update Q-functions by one step of gradient descent
      # pdb.set_trace()
      value_loss = (critic_1(batch['state'], batch['action']) - y_q).pow(2).mean() + (critic_2(batch['state'], batch['action']) - y_q).pow(2).mean()
      critics_optimiser.zero_grad()
      value_loss.backward()
      critics_optimiser.step()

      # Update V-function by one step of gradient descent
      value_loss = (value_critic(batch['state']) - y_v).pow(2).mean()
      value_critic_optimiser.zero_grad()
      value_loss.backward()
      value_critic_optimiser.step()

      # Update policy by one step of gradient ascent
      policy_loss = (weighted_sample_entropy - critic_1(batch['state'], action)).mean()
      actor_optimiser.zero_grad()
      if (policy_loss != policy_loss).all():
        #check if any values are NaN
        print("NaN found")
        pass
      else:
        try: 
          policy_loss.backward()
          actor_optimiser.step()
        except:
          print(policy_loss)
          pdb.set_trace()

      # Update target value network
      update_target_network(value_critic, target_value_critic, POLYAK_FACTOR)

    if step > UPDATE_START and step % TEST_INTERVAL == 0:
      actor.eval()
      total_reward = test(actor)
      total_transfer_reward = test_transfer(actor, path)
      # if not MULTIPROCESSING:
      pbar.set_description('Step: %i | Reward: %f | Transfer Reward: %f' % (step, total_reward, total_transfer_reward))
      steps.append(step)
      rewards.append(total_reward)
      transfer_rewards.append(total_transfer_reward)
      # plot(step, total_reward, 'sac' + ('_penalty' if penalty else ''))
      actor.train()
  return rewards, transfer_rewards



def objective(trial):
  epsilon = trial.suggest_loguniform('epsilon', .01, .1)
  actor_scaling = trial.suggest_loguniform('actor_scaling', .5, 2.)
  critic_lr = trial.suggest_loguniform('critic_lr', LEARNING_RATE*.01, LEARNING_RATE*3)
  log_hidden_size = trial.suggest_int('log_hidden_size',5, 9)

  return -sum(train_td3(penalty=True, epsilon=epsilon, actor_LEARNING_RATE=critic_lr*actor_scaling, critic_LEARNING_RATE=critic_lr, h=2**log_hidden_size)[0])

def hyperparameter_search():
  study = optuna.create_study()
  study.optimize(objective, n_trials=100)
  print(study.trials)
  print(study.best_params)



def evaluate(penalty=True):
  samples = SAMPLES
  train_rewards = []
  test_rewards = []
  title = 'td3_'+env_name+('_penalty' if penalty else '')
  if args.algorithm=='td3':
    train_alg=train_td3
  elif args.algorithm=='sac':
    train_alg=train_sac
  else:
    print('training algorithm not found')
    assert False

  if MULTIPROCESSING:
    p = mp.Pool(20)
    # reward_list = print(p.map(train_td3, [(penalty, .025)]*samples))
    reward_list = print(p.map(train_alg, [(penalty, .025)]*samples))
    train_rewards = [i[0] for i in reward_list]
    test_rewards = [i[1] for i in reward_list]
    with open('data/' + title + '.pkl', 'wb+') as f:
      pickle.dump((train_rewards, test_rewards), f)
  else:
    for i in range(samples):
    
      # rewards, transfer_rewards= train_td3(penalty=penalty)#, epsilon=.05)
      rewards, transfer_rewards= train_alg(penalty=penalty)#, epsilon=.05)
      # rewards = transfer_rewards = [random.random() for i in range(UPDATE_START, MAX_STEPS,TEST_INTERVAL)]
      try:
        with open('data/' + title + '.pkl', 'rb+') as f:
          train_rewards, test_rewards = pickle.load(f)
      except:
        print('Unable to find data file. Making new one')

      train_rewards.append(rewards)
      test_rewards.append(transfer_rewards)
      with open('data/' + title + '.pkl', 'wb+') as f:
        pickle.dump((train_rewards, test_rewards), f)
    

  #recursive averaging
  ave = lambda tr: np.mean(np.array(tr))
  print(title)
  print(ave(train_rewards))

  print(title)
  print(ave(test_rewards))

  return train_rewards, test_rewards

def compare():
  steps = [i for i in range(UPDATE_START, MAX_STEPS,TEST_INTERVAL)]
  steps_list = [steps]*2

  p_train, p_test = evaluate(penalty=True)
  no_p_train, no_p_test =evaluate(penalty=False)
  # no_p_test= [[random.random() for step in steps]for s in range(SAMPLES)] 
  # no_p_train=[[random.random() for step in steps]for s in range(SAMPLES)] 
  # p_test=[[random.random() for step in steps]for s in range(SAMPLES)] 
  # p_train = [[random.random() for step in steps]for s in range(SAMPLES)] 

  train_rewards = [p_train, no_p_train]
  test_rewards = [p_test, no_p_test]
  train_rewards = np.array([p_train, no_p_train])
  test_rewards = np.array([p_test, no_p_test])
  labels = ['Our method', 'Baseline']
  # pdb.set_trace()
  multiplot(steps_list, train_rewards, labels, 'td3 '+env_name+' same environment')
  multiplot(steps_list, test_rewards, labels, 'td3 '+env_name+' modified environment')



SAMPLES = 10

env_dict = {}
#env_dict['swimmer'] = ('Swimmer-v3', 'mod_envs/swimmer/', 8, 2)
#env_dict['pendulum'] = ('Pendulum-v0', None, 3, 1)
env_dict['hopper'] = ('Hopper-v3', 'mod_envs/hopper/', 11, 3)
env_dict['halfcheetah'] = ('HalfCheetah-v3', 'mod_envs/halfcheetah/' , 17, 6)
env_dict['ant'] = ('Ant-v3', 'mod_envs/ant/', 111, 8)
env_dict['walker'] = ('Walker2d-v3', 'mod_envs/walker/', 17, 6)
env_dict['humanoid'] = ('Humanoid-v3', 'mod_envs/humanoid/', 376, 17)

from randomize_xml import randomize_xml
for desc in env_dict.values():
  path = desc[1]
  if path is not None:
    filenames = [os.getcwd() + '/'+ path + xml for xml in os.listdir(path) if xml.endswith('.xml') ]
    fn = sorted(filenames, key = lambda x: len(x))
    xml_file = fn[0]
    print(fn[0])
    randomize_xml(xml_file, scale=0.3, count=4)

if args.task =='':
  # OPTIMZE = True
  OPTIMZE = False
  for task in env_dict.keys():
  # for task in reversed(list(env_dict.keys())):
    env_name, path, state_dim, action_dim = env_dict[task]
    if OPTIMZE:
      path=None
      hyperparameter_search()
    else:
      compare()
else:
  env_name, path, state_dim, action_dim = env_dict[args.task]
  print(env_dict[args.task])
  evaluate(args.no_penalty)
  # load_and_plot()