from agent import PosePPO
from trainer import PoseTrainer
# from srg.agents.trainer import Trainer
from srg.envs.bullet3.gyms import PoseEnv 
from srg.agents.utilities.data_structures.Config import Config

import os
import sys
import time
from os.path import dirname, abspath
curr_dir = os.path.dirname(os.path.realpath(__file__))

config = Config()
config.seed = 1
# config.environment = gym.make("CartPole-v0")
config.num_episodes_to_run = 20
config.file_to_save_data_results = os.path.join(curr_dir, "results/data.pkl")
config.file_to_save_results_graph = os.path.join(curr_dir, "results/graph.png")
config.file_to_save_learnt_model = os.path.join(curr_dir, "results/model.pt")
config.show_solution_score = False
config.visualise_individual_results = True
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = True
config.randomise_random_seed = True
config.save_model = True

config.hyperparameters = {
    "Policy_Gradient_Agents": {
        "learning_rate": 0.05,
        "linear_hidden_units": [50, 50, 50],
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 5,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 1,
        "normalise_rewards": True,
        "gradient_clipping_norm": 7.0,
        "mu": 0.0, #only required for continuous action games
        "theta": 0.0, #only required for continuous action games
        "sigma": 0.0, #only required for continuous action games
        "epsilon_decay_rate_denominator": 1.0,
        "clip_rewards": False } }

if __name__ == '__main__':
    config.environment = PoseEnv()
    agents = [PosePPO]
    trainer = PoseTrainer(config, agents)
    trainer.run_games_for_agents()