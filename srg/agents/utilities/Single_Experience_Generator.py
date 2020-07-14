import random
import torch
import sys
# from contextlib import closing
#
# from pathos.multiprocessing import ProcessingPool as Pool

# from torch.multiprocessing import Pool
from random import randint

from srg.agents.utilities.OU_Noise import OU_Noise
from srg.agents.utilities.Utility_Functions import create_actor_distribution

class Single_Experience_Generator(object):
    """ Plays n episode in parallel using a fixed agent. Only works for PPO or DDPG type agents at the moment, not Q-learning agents"""

    def __init__(self, environment, policy, seed, hyperparameters, action_size, use_GPU=False, action_choice_output_columns=None):
        self.use_GPU = use_GPU
        self.environment =  environment
        self.action_types = "DISCRETE" if self.environment.action_space.dtype == int  else "CONTINUOUS"
        self.action_size = action_size
        self.policy = policy
        self.action_choice_output_columns = action_choice_output_columns
        self.hyperparameters = hyperparameters
        if self.action_types == "CONTINUOUS": 
            self.noise = OU_Noise(self.action_size, seed, 
                                  self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], 
                                  self.hyperparameters["sigma"] )

    def play_n_episodes(self, n, exploration_epsilon=None):
        """Plays n episodes in sequential using the fixed policy and returns the data"""
        self.exploration_epsilon = exploration_epsilon
        states_for_all_episodes = []
        actions_for_all_episodes = []
        rewards_for_all_episodes = []
        for _ in range(n):
            episode = self(exploration_epsilon)
            states_for_all_episodes.append(episode[0])
            actions_for_all_episodes.append(episode[1])
            rewards_for_all_episodes.append(episode[2])
        return states_for_all_episodes, actions_for_all_episodes, rewards_for_all_episodes

    def __call__(self, n):
        exploration = max(0.0, random.uniform(self.exploration_epsilon / 3.0, self.exploration_epsilon * 3.0))
        return self.play_1_episode(exploration)

    def play_1_episode(self, epsilon_exploration):
        """Plays 1 episode using the fixed policy and returns the data"""
        state = self.reset_game()
        done = False
        episode_states = []
        episode_actions = []
        episode_rewards = []
        while not done:
            action = self.pick_action(self.policy, state, epsilon_exploration)
            next_state, reward, done, _ = self.environment.step(action)
            if self.hyperparameters["clip_rewards"]: 
                reward = max(min(reward, 1.0), -1.0)
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state
        return episode_states, episode_actions, episode_rewards

    def reset_game(self):
        """Resets the game environment so it is ready to play a new episode"""
        seed = randint(0, sys.maxsize)
        torch.manual_seed(seed) # Need to do this otherwise each worker generates same experience
        state = self.environment.reset()
        if self.action_types == "CONTINUOUS": 
            self.noise.reset()
        return state

    PI = 3.1415026
    def pick_action(self, policy, state, epsilon_exploration=None):
        """Picks an action using the policy"""
        if self.action_types == "DISCRETE":
            if random.random() <= epsilon_exploration:
                action = random.randint(0, self.action_size - 1)
                return action

        state = torch.from_numpy(state).float().unsqueeze(0)
        actor_output = policy.forward(state) # FIX IT
        # means = torch.clamp(actor_output[:,0:self.environment.get_action_size()], min=-1000, max=1000)
        # stds = torch.clamp(actor_output[:, self.environment.get_action_size():], min=-5,max=5)

        PI = 3.1415026
        means = torch.clamp(actor_output[:,0:self.environment.get_action_size()], min=-PI, max=PI)
        stds = torch.clamp(actor_output[:, self.environment.get_action_size():], min=-PI/20,max=PI/20)
 
        # stds = log_stds.exp()
        actor_output = torch.cat((means, stds),1)
        
        if self.action_choice_output_columns is not None:
            actor_output = actor_output[:, self.action_choice_output_columns]
        action_distribution = create_actor_distribution(self.action_types, actor_output, self.action_size)
        action = action_distribution.sample().cpu()

        if self.action_types == "CONTINUOUS": 
            action += torch.Tensor(self.noise.sample())
        else: 
            action = action.item()
        return action