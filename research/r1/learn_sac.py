from loc_agent_sac import PoseSAC
from loc_trainer import PoseTrainer
# from srg.agents.trainer import Trainer
from srg.envs.bullet3.gyms import PoseEnv
# from srg.agents.actor_critic_agents.sac import SAC 
from srg.agents.utilities.data_structures.Config import Config
from torch.distributions.normal import Normal
import os
import sys
import time
from os.path import dirname, abspath
import torch
import pybullet as bullet

curr_dir = os.path.dirname(os.path.realpath(__file__))

config = Config()
config.seed = 1
config.runs_per_agent = 3
config.num_episodes_to_run = 50000
config.file_to_save_data_results = os.path.join(curr_dir, "results/data_sac.pkl")
config.file_to_save_results_graph = os.path.join(curr_dir, "results/graph_sac.png")
config.file_to_save_learnt_model = os.path.join(curr_dir, "results/model_sac.pt")
config.file_to_save_training_log = os.path.join(curr_dir, "results/training_sac.log")
config.show_solution_score = False
config.visualise_individual_results = True
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.use_GPU = True
config.overwrite_existing_results_file = True
config.randomise_random_seed = True
config.save_model = True
config.load_previous_model = False

config.hyperparameters = {
    "Actor_Critic_Agents": {
        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64, 32],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier" },
        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64, 32],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier" },
        "min_steps_before_learning": 400,
        # "batch_size": 256,
        "batch_size": 4,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True,
        "clip_rewards": False } }

if __name__ == '__main__':

    # Train the agent
    env = PoseEnv()
    config.environment = env
    agents = [PoseSAC]
    trainer = PoseTrainer(config, agents)
    trainer.run_games_for_agents()

    # Test the learnt polcy
    state = env.reset()
    policy = torch.load(config.file_to_save_learnt_model).cuda()

    try:
        # flag = True
        # x_in = bullet.addUserDebugParameter("x", -2, 2, 0)
        # y_in = bullet.addUserDebugParameter("y", -2, 2, 0)
        # z_in = bullet.addUserDebugParameter("z", 0.5, 2, 1)
        # i = 0
        while True:
            state = torch.tensor(state).cuda().float().reshape(1,-1)
            pi_dist = policy(state)
            mean, std = pi_dist[0,0:env.get_action_size()], pi_dist[0,env.get_action_size():]
            distribution = Normal(mean, std) 
            action = distribution.sample()
            action = torch.clamp(action, min= -500, max=500)
            next_state, *_ = env.step(action.cpu())
            state = next_state
            s_dict = env.robot.get_state(form='dict_like') 
            pos = s_dict['end_effector']['link_world_pos']
            print("x = {d[0]}, y = {d[1]}, z = {d[2]}".format(d=pos))

        bullet.disconnect()
    except:
        bullet.disconnect()