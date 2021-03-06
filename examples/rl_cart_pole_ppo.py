import os
import sys
from os.path import dirname, abspath
import torch
import gym
from srg.agents.utilities.data_structures.Config import Config
from srg.agents.trainer import Trainer
from srg.agents.policy_gradient_agents.ppo import PPO


curr_dir = os.path.dirname(os.path.realpath(__file__))

config = Config()
config.seed = 1
config.environment = gym.make("CartPole-v0")
config.num_episodes_to_run = 400
config.file_to_save_data_results = os.path.join(curr_dir, "results_rl_cart_pole/data_ppo.pkl")
config.file_to_save_results_graph = os.path.join(curr_dir, "results_rl_cart_pole/graph_ppo.png")
config.file_to_save_learnt_model = os.path.join(curr_dir, "results_rl_cart_pole/policy_ppo.pt")
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
        "linear_hidden_units": [20, 20],
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 5,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 4,
        "normalise_rewards": True,
        "gradient_clipping_norm": 7.0,
        "mu": 0.0, #only required for continuous action games
        "theta": 0.0, #only required for continuous action games
        "sigma": 0.0, #only required for continuous action games
        "epsilon_decay_rate_denominator": 1.0,
        "clip_rewards": False } }


if __name__ == "__main__":
    # Train a PPO policy
    torch.multiprocessing.set_start_method("spawn")
    agents = [PPO]
    trainer = Trainer(config, agents)
    trainer.run_games_for_agents()
    
    # Test the learnt policy
    env = gym.make('CartPole-v0')
    policy = torch.load(config.file_to_save_learnt_model).cuda()
    ob = env.reset()
    for i in range(1000):
        env.render()
        ob_t = torch.tensor(ob, dtype=torch.float32).cuda()
        action = policy(ob_t.reshape([1,-1])).cpu().reshape(-1)
        # action[0] and action [1] probabilities
        if action[0] < action[1]:
            ob, _, _, _ = env.step(1)
        else:
            ob, _, _, _ = env.step(0)
    env.close()




