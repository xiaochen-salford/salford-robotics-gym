from ..robot_env import RobotEnv

import numpy as np
import pybullet as bullet
from gym import spaces
from gym.spaces import Box
from collections import namedtuple
import os


curr_dir = os.path.dirname(os.path.realpath(__file__))


class RobotUR5RobotiqC2_Config:
  def __init__(self):
    self.model_path = curr_dir + '/../models/mdl_ur5robotiqc2.urdf'

    self.joint_types = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"] 
    self.joint_info = namedtuple\
    (
      "joint_info", 
      [
        "idx", "name", "type", "lower_limit", "upper_limit", "max_force", "max_velocity",
        "controllable"
      ]
    ) 
    
    self.orderred_joints = \
    [
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint", 
      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
      "robotiq_85_left_knuckle_joint" 
    ]

    self.orderred_joint_indices = range(len(self.orderred_joints))


# WIP
class RobotUR5RobotiqC2(RobotEnv, RobotUR5RobotiqC2_Config):
  def __init__(self, initial_pos=[0,0,0], 
      inital_orn=bullet.getQuaternionFromAxisAngle([0,0,0]), time_step=0.01):
    super().__init__(self.model_path, initial_pos, inital_orn, time_step=time_step)
    
    self.arm_link_num = 8
    self.gripper_link_num = 10
    self.arm_dof_num = 6
    self.dof_num = self.arm_dof_num + self.gripper_dof_num
    self.gripper_dof_num = 6
    self.arm_idx_range = range(0, self.arm_dof_num)
    self.griper_idx_range = range(self.arm_dof_num, self.arm_link_num+self.gripper_dof_num)

    self.joint_indices = range(self.arm_dof_num+self.gripper_dof_num)
    self.init_action_space() 
    self.init_observation_space() 

    bullet.enableJointForceTorqueSensor(self.id, range(self.dof_num), enableSensor=True)

  #TODO
  def init_action_space(self):
    arm = dict()
    arm['joint_torque_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_dof_num,1), dtype=np.float32)
    
    gripper = dict()
    griiper['joint_torque_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_dof_num,1), dtype=np.float32)

    super().action['arm'] = arm
    super().action['gripper'] = gripper 

  def ext_action_space(self):
    """Extend action space kes for individual gym envs later"""
    raise NotImplementedError

  def init_observation_space(self):
    arm = dict()
    arm['joint_pos_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_dof_num,3), dtype=np.float32)
    arm['jonit_vel_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_dot_num,3), dtype=np.float32)
    arm['link_glb_pos_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_link_num,3), dtype=np.float32)
    arm['link_glb_orn_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_link_num,4), dtype=np.float32)
    arm['link_loc_pos_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_link_num,3), dtype=np.float32)
    arm['link_loc_orn_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_link_num,4), dtype=np.float32)

    gripper = dict()
    griiper['joint_pos_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_dof_num,3), dtype=np.float32)
    griiper['jonit_vel_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_dot_num,3), dtype=np.float32)
    griiper['link_glb_pos_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_link_num,3), dtype=np.float32)
    griiper['link_glb_orn_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_link_num,4), dtype=np.float32)
    griiper['link_loc_pos_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_link_num,3), dtype=np.float32)
    griiper['link_loc_orn_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_link_num,4), dtype=np.float32)

    super().observation['arm'] = arm 
    super().observation['gripper'] = gripper 

  def init_observation_space(self):
    """Extend observation space keys for individual gym envs later"""
    raise NotImplementedError
  
  def set_action(self, action):
    """Use torque control"""
    bullet.setJointMotorControlArray(self.id, self.joint_indices, bullet.TORQUE_CONTROL, force=action)

  def get_observation(self):
    joint_states = np.array(bullet.getJointStates(self.id, self.orderred_joint_indices))

    super().observation['arm']['joint_poss'] = np.array([joint_states[idx][0] for idx in self.arm_idx_range])
    super().observation['arm']['joint_vels'] = np.array([joint_states[idx][1] for idx in self.arm_idx_range])
    super().observation['arm']['joint_rctfs'] = np.array([joint_states[idx][2] for idx in self.arm_idx_range])
    super().observation['arm']['joint_actfs'] = np.array([joint_states[idx][3] for idx in self.arm_idx_range])

    super().observation['gripper']['joint_poss'] = np.array([joint_states[idx][0] for idx in self.gripper_idx_range])
    super().observation['gripper']['joint_vels'] = np.array([joint_states[idx][1] for idx in self.gripper_idx_range])
    super().observation['gripper']['joint_rctfs'] = np.array([joint_states[idx][2] for idx in self.gripper_idx_range])
    super().observation['gripper']['joint_actfs'] = np.array([joint_states[idx][3] for idx in self.gripper_idx_range])

    #TODO: get link state

    #TODO: get link contacts
  