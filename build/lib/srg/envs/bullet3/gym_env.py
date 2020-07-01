import numpy as np
from .robot_env import RobotEnv
from . import rotations, robot_env, utils

def goal_distance(goal_a, goal_b):
  assert goal_a.shape == goal_b.shape
  return np.linalg.norm(goal_a - goal_b, axis=-1)


class GymEnv:
  """
  Superclass for all Salford Robotics Gym environments.
  GymEnv is designed to interact with Agent direclty. A gym env should consist of one robot env, while other objects 
  are defined right under the inherenet env class. 
  """
  def __init__(self):
    self.robot = None

  def step(self, action):
    raise NotImplementedError

  def compute_reward(self, achieved_goal, goal, info):
    raise NotImplementedError

  def reset(self):
    raise NotImplementedError