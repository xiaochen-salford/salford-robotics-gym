import os
import copy
import numpy as np

import srg
from srg import error, spaces
from srg.utils import seeding

try:
    import pybullet as bullet
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pybullet.".format(e))

DEFAULT_SIZE = 500

# working in progress...
class RobotEnv:
  """
  Superclass for all robots 
  RobotEnv is sub-env for GymEnv; it is used to specify a robot and sould be attached to RobotEnv
  """
  def __init__(self, model_path, initial_pos, inital_orn, time_step=0.01):
    if model_path.startswith('/'):
      fullpath = model_path
    else:
      fullpath = os.path.join(os.path.dirname(__file__), 'models', model_path)
    if not os.path.exists(fullpath):
      raise IOError('File {} does not found'.format(fullpath))

    self.id = bullet.loadURDF(fullpath, initial_pos, inital_orn, flags=bullet.URDF_USE_INERTIA_FROM_FILE)
    self.time_step = time_step

    self.action = dict()
    self.observation = dict()
    self.action_space_info = None
    self.observation_space_info = None
    
  def init_action_space(self):
    raise NotImplementedError
  
  def init_observation_space(self):
    raise NotImplementedError

  def set_action(self):
    raise NotImplementedError
  
  def get_observation(self):
    raise NotImplementedError
