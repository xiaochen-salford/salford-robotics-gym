import numpy as np
from .robot_env import RobotEnv
from . import rotations, robot_env, utils

class GymEnv:
    """ 
    The bsae class for all Salford Robotics Gym environments.
    GymEnv is designed to interact with Agent direclty. A gym env should consist of one robot env, while other objects 
    are defined right under the inherenet env class. 
    """

    def __init__(self):
      self.robot = None

    def step(self, action, goal=None):
      raise NotImplementedError

    def compute_reward(self, goal, achieved_goal, info):
      raise NotImplementedError

    def reset(self):
      raise NotImplementedError

    def init_action_space(self):
      raise NotImplementedError

    def init_state_space(self):
      raise NotImplementedError

    def init_action_size(self):
      raise NotImplementedError

    def init_state_size(self):
      raise NotImplementedError

