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
        # self.model_path = curr_dir + '/../models/mdl_ur5robotiqc2.urdf'

        self.joint_types = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"] 

        self.joint_info = namedtuple(
                "joint_info", 
                ["idx", 
                 "name", 
                 "type", 
                 "lower_limit", "upper_limit", 
                 "max_force", "max_velocity", 
                 "controllable"])

        self.orderred_joints = ["shoulder_pan_joint", 
                                "shoulder_lift_joint", 
                                "elbow_joint", 
                                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", 
                                "robotiq_85_left_knuckle_joint"]

        self.orderred_joint_indices = range(len(self.orderred_joints))

        self.joint_dim_tmpl = {
                'joint_state_channels': ['position', 
                                         'velocity', 
                                         'constrained force', 
                                         'applied torque'],
                'joint_state_num': None,
                'joint_state_channels_dims': [1, 1, 6, 1]}


# WIP
class RobotUR5RobotiqC2(RobotEnv, RobotUR5RobotiqC2_Config):
    def __init__(self, initial_pos=[0,0,0], 
        inital_orn=bullet.getQuaternionFromEuler([0,0,0]), time_step=0.01):
        self.model_path = curr_dir + '/../models/mdl_ur5robotiqc2.urdf'

        self.arm_link_num = 8
        self.gripper_link_num = 10
        self.arm_dof_num = 6
        self.gripper_dof_num = 1
        self.dof_num = self.arm_dof_num + self.gripper_dof_num
        self.arm_idx_range = range(0, self.arm_dof_num)
        self.griper_idx_range = range(self.arm_dof_num, self.arm_link_num+self.gripper_dof_num)

        self.joint_indices = range(self.arm_dof_num+self.gripper_dof_num)
        
        super().__init__(self.model_path, initial_pos, inital_orn, time_step=time_step)
        self.ext_action_space
        self.ext_observation_space

    #TODO
    def init_action_space(self):
        arm = dict()
        arm['joint_torque_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_dof_num,), dtype=np.float32)
        
        gripper = dict()
        gripper['joint_torque_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_dof_num,), dtype=np.float32)

        self.action.update({'arm':arm})
        self.action.update({'gripper': gripper}) 

    def ext_action_space(self):
        """Extend action space kes for individual gym envs later"""
        pass

    def init_observation_space(self):
        arm = dict()
        arm['joint_pos_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_dof_num,3), dtype=np.float32)
        arm['jonit_vel_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_dof_num,3), dtype=np.float32)
        arm['link_glb_pos_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_link_num,3), dtype=np.float32)
        arm['link_glb_orn_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_link_num,4), dtype=np.float32)
        arm['link_loc_pos_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_link_num,3), dtype=np.float32)
        arm['link_loc_orn_list'] = Box(low=-np.inf, high=np.inf, shape=(self.arm_link_num,4), dtype=np.float32)

        gripper = dict()
        gripper['joint_pos_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_dof_num,3), dtype=np.float32)
        gripper['jonit_vel_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_dof_num,3), dtype=np.float32)
        gripper['link_glb_pos_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_link_num,3), dtype=np.float32)
        gripper['link_glb_orn_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_link_num,4), dtype=np.float32)
        gripper['link_loc_pos_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_link_num,3), dtype=np.float32)
        gripper['link_loc_orn_list'] = Box(low=-np.inf, high=np.inf, shape=(self.gripper_link_num,4), dtype=np.float32)

        self.observation['arm'] = arm 
        self.observation['gripper'] = gripper 

    def ext_observation_space(self):
        """Extend observation space keys for individual gym envs later"""
        pass

    def set_action(self, action):
        """Use torque control"""
        bullet.setJointMotorControlArray(self.id, range(self.dof_num), bullet.TORQUE_CONTROL, forces=np.zeros(self.dof_num))
        # bullet.setJointMotorControl2(self.id, self.joint_indices, bullet.TORQUE_CONTROL, 0)
        # bullet.setJointMotorControl2(self.id, 1, bullet.TORQUE_CONTROL, force=-1000)
        # bullet.setJointMotorControl2(self.id, 2, bullet.TORQUE_CONTROL, force=-1000)

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

    # TODO 
    def init_action_dim(self):
        self.action_dim['arm_dim'] = self.arm_dof_num
        self.action_dim['gripper_dim'] = self.gripper_dof_num  

    # TODO 
    def init_observation_dim(self):
        arm_joints = self.joint_dim_tmpl.copy()
        arm_joints['joint_state_num'] = self.arm_dof_num

        gripper_joints = self.joint_dim_tmpl.copy() 
        gripper_joints['joint_state_num'] = self.gripper_dof_num     
        
        self.observation_dim['arm_joints'] = arm_joints
        self.observation_dim['gripper_joints'] = gripper_joints
    