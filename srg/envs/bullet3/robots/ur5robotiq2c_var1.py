from srg.envs.bullet3.robot_env import RobotEnv
import numpy as np
import pybullet as bullet
from gym import spaces
from gym.spaces import Box, Dict
from collections import namedtuple
import os
from os import path
from collections import OrderedDict

curr_dir = os.path.dirname(os.path.realpath(__file__))

class RobotUR5RobotiqC2Config:
    model_path = path.normpath(path.join(curr_dir, '../models/mdl_ur5robotiqc2.urdf'))
    joint_types = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    ordered_joints = (
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint',
        'ee_fixed_joint',
        'arm_gripper_joint',
        'robotiq_85_base_joint',
        'robotiq_85_left_knuckle_joint',
        'robotiq_85_left_finger_joint',
        'robotiq_85_right_knuckle_joint',
        'robotiq_85_right_finger_joint',
        'robotiq_85_left_inner_knuckle_joint',
        'robotiq_85_left_finger_tip_joint',
        'robotiq_85_right_inner_knuckle_joint',
        'robotiq_85_right_finger_tip_piont')
    ordered_links = (
        'base_link',
        'shoulder_link',
        'upper_arm_link',
        'forarm_link',
        'wrist_1_link',
        'wrist_2_link',
        'wrist_3_link',
        'ee_link',
        'robotiq_85_adapter_link',
        'robotiq_85_base_link',
        'robotiq_85_left_knuckle_link',
        'robotiq_85_left_finger_link',
        'robotiq_85_right_knuckle_link',
        'robotiq_85_right_finger_link',
        'robotiq_85_left_inner_knuckle_link',
        'robotiq_85_left_finger_tip_link',
        'robotiq_85_right_inner_knuckle_link',
        'robotiq_85_right_finger_tip' )

    active_joint_indices = [1, 2, 3, 4, 5, 6, 10, 11, 12, 14, 16, 17]
    arm_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7] # The 7th joint is fixed with the gripper
    arm_active_joint_indices = [1, 2, 3, 4, 5, 6]
    gripper_joint_indices = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    gripper_active_joint_indices = [10, 11, 12, 14, 16, 17]
    arm_link_indices = [0, 1, 2, 3, 4, 5]
    gripper_link_indices = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  

    def __init__(self):
        self.joints_num = len(type(self).ordered_joints)
        self.joint_indices = OrderedDict([(type(self).ordered_joints[i], i) for i in range(self.joints_num)])

        self.links_num = len(type(self).ordered_links)
        self.link_indices = OrderedDict([(type(self).ordered_links[i], i) for i in range(self.links_num)])

        self.active_joint_num = len(type(self).active_joint_indices)

        self.arm_joints_num = len(type(self).arm_joint_indices)
        self.gripper_joints_num = len(type(self).gripper_joint_indices)

        self.arm_active_joints_num = len(type(self).arm_active_joint_indices)
        self.gripper_active_joints_num = len(type(self).gripper_active_joint_indices)

        self.arm_links_num = len(type(self).arm_link_indices)
        self.gripper_links_num = len(type(self).gripper_link_indices)


# working in progress...
class RobotUR5RobotiqC2_Var1(RobotEnv):

    def __init__(self, initial_pos=[0,0,0], 
                 initial_orn=bullet.getQuaternionFromEuler([0,0,0])):
        self.config = RobotUR5RobotiqC2Config()
        RobotEnv.__init__(self, self.config, initial_pos, initial_orn)
        for idx in range(self.config.joints_num):
            bullet.enableJointForceTorqueSensor(self.id, idx, enableSensor=True)
        self.curr_action = np.zeros(self.config.arm_active_joints_num)
        self.prev_action = np.zeros(self.config.arm_active_joints_num)
        self.ext_action_space()
        self.ext_state_space()

    def init_action_space(self):
        arm_active_joints_num = self.config.arm_active_joints_num
        return Dict({
            'arm': Dict({'joint_torques': Box(low=-np.inf, high=np.inf, shape=(arm_active_joints_num,), dtype=np.float32)}) })  

    def init_state_space(self):
        arm_joints_num = self.config.arm_joints_num
        arm_links_num = self.config.arm_links_num
        return Dict({
            'arm': Dict({
                'joint_poss': Box(low=-np.inf, high=np.inf, shape=(arm_joints_num,3), dtype=np.float32),
                'joint_vels': Box(low=-np.inf, high=np.inf, shape=(arm_joints_num,3), dtype=np.float32),
                'joint_rfs': Box(low=-np.inf, high=np.inf, shape=(arm_joints_num,6), dtype=np.float32), 
                'joint_prev_afs': Box(low=-np.inf, high=np.inf, shape=(arm_joints_num,1), dtype=np.float32)}),

            'end_effector': Dict({ 
                'link_pos': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32) }) })
    
    def get_action_size(self):
        return self.config.arm_active_joints_num

    def get_state_size(self):
        """
        size = curr_ee_world_pos + arm_active_joints_poss + arm_active_joint_vels + arm_active_joint_rfs + prev_appliced_torques  
        """
        n1 = self.config.arm_joints_num
        n2 = self.config.arm_active_joints_num
        return 3 + n1 + n1 + 6*n1 + n2    

    def ext_action_space(self):
        """Extend action space if needed"""
        pass

    def ext_state_space(self):
        """Extend observation space if needed"""
        pass

    def set_action(self, action, form='flat'):
        """
        action: {'arm': ..., 'gripper': ...} or 1-D array
        """
        if form == 'dict_like':
            assert 'arm' in action and action['arm'] is not None
            bullet.setJointMotorControlArray(self.id, 
                                             self.config.arm_active_joint_indices, 
                                             bullet.TORQUE_CONTROL, 
                                             forces=action['arm'] )
        else : # form == 'flat'
            assert len(action) == self.config.arm_active_joints_num, 'We only concern controlling the arm at currently'
            assert action.shape == (len(action), )
            bullet.setJointMotorControlArray(self.id, 
                                             self.config.arm_active_joint_indices, 
                                             bullet.POSITION_CONTROL, 
                                             targetPositions=action.tolist() )
            self.prev_action = self.curr_action
            self.curr_action = action

    def get_state(self, form='flat'):
        if form == 'dict_like': 
            arm_state = {}
            arm_state['joint_poss'] = bullet.getJointStates(self.id, self.config.arm_joint_indices)
            gripper_state = {}
            gripper_state['joint_poss'] = bullet.getJointStates(self.id, self.config.gripper_joint_indices)
            ee_idx = self.config.ordered_links.index('ee_link')
            ee_state = bullet.getLinkState(self.id, ee_idx)
            end_effector_state = {}
            end_effector_state['link_world_pos'] = ee_state[0]
            end_effector_state['link_world_orn'] = ee_state[1]
            return {'arm': arm_state,
                    'gripper': gripper_state,
                    'end_effector': end_effector_state }
        else: # form == 'flat'
            arm_state = bullet.getJointStates(self.id, self.config.arm_joint_indices)
            arm_joint_poss = [state[0] for state in arm_state]
            arm_joint_vels = [state[1] for state in arm_state]
            arm_joint_rfs = [rf for state in arm_state for rf in state[2]] # 6-dim refaction force for each joint
            ee_idx = self.config.ordered_links.index('ee_link')
            ee_state = bullet.getLinkState(self.id, ee_idx)
            ee_pos = [e for e in ee_state[0]] 
            state = np.array(ee_pos + arm_joint_poss + arm_joint_vels + arm_joint_rfs)
            state = np.append(state, self.prev_action) 
            return state
    
    def get_goal_state(self):
        """Get the end-effector position"""
        ee_idx = self.config.ordered_links.index('ee_link')
        ee_state = bullet.getLinkState(self.id, ee_idx)
        ee_pos = np.array(ee_state[0])
        return ee_pos
        
    def reset(self):
        for idx in self.config.active_joint_indices:
            bullet.resetJointState(self.id, idx, 0.0)
        return self.get_state()

