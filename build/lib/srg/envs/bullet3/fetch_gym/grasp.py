import os
from srg import utils
from ..gym_env import GymEnv 
from ..robots import RobotUR5RobotiqC2
import pybullet as bullet

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')

class GraspEnv(GymEnv):
  #TODO
  def __init__(self):
    self.robot = RobotUR5RobotiqC2()
    self.plane = bullet.loadURDF()  

    #TODO: load a block object
    pass

  #TODO 
  def step(self, action):
    self.robot.set_action(action)
    bullet.stepSimulation()
    observation = self.robot.get_observation()
    reward = self.compute_reward()
    return observation, reward 

  #TODO 
  def reset(self):
    pass

  #TODO 
  def compute_reward(self, achieved_goal, goal, info):
    reward = 1
    return reward
