import os
from srg import utils
from ..gym_env import GymEnv 
from ..robots import RobotUR5RobotiqC2
import pybullet as bullet
import numpy as np
from numpy.linalg import norm
from gym.utils import seeding
from gym.spaces import Box, Dict

class PoseEnv(GymEnv):

    def __init__(self):
        GymEnv.__init__(self)
        self.client_id = bullet.connect(bullet.GUI)
        bullet.setTimeStep(0.01)
        self.robot = RobotUR5RobotiqC2()
        self.ee_goal_pos = np.array([1.0, 1.0, 1.0])
        self.env_state = self.ee_goal_pos 
        self.goal_state_size = self.ee_goal_pos.size
        self.action_size = self.get_action_size()
        self.state_size = self.get_state_size()
        self.action_space = self.init_action_space()
        self.state_space = self.init_state_space()

        self.buffer_length = 500
        self.buffer = np.zeros(self.buffer_length)
        self.steps_num = 0
        self.step_stop_num = 1000
        self.seed()
    
    def init_action_space(self):
        return self.robot.action_space
    
    def init_state_space(self):
        state_space = self.robot.state_space
        state_space.spaces['goal'] = Dict({
            'end_effector_pos': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32) })
        return state_space

    def get_action_size(self):
        return self.robot.get_action_size()

    def get_state_size(self):
        return self.robot.get_state_size() + self.goal_state_size
    
    def get_goal_state(self):
        return self.robot.get_goal_state()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #working in progress... 
    def step(self, action, goal=None):
        self.robot.set_action(action)
        bullet.stepSimulation()
        next_state = self.robot.get_state()
        next_state = np.append(self.ee_goal_pos, next_state)
        self.steps_num += 1

        if self.steps_num >= self.step_stop_num: done = True
        else: done = False 

        if goal is not None:
            achieved_goal = self.get_goal_state()
            reward = self.compute_reward(goal, achieved_goal)
            return next_state, reward, done, {}
        else: # Use default goal
            achieved_goal = self.get_goal_state()
            reward = self.compute_reward(self.ee_goal_pos, achieved_goal)
            return next_state, reward, done, {}
        
    def reset(self):
        self.steps_num = 0
        bullet.resetSimulation()
        # bullet.connect(bullet.GUI)
        # bullet.setPhysicsEngineParameter(numSolverIterations=150)
        delattr(self, 'robot')
        self.robot = RobotUR5RobotiqC2()

        init_state = self.robot.get_state()
        init_state = np.append(self.ee_goal_pos, init_state) 
        return init_state

    def compute_reward(self, goal=None, achieved_goal=None, info=None):
        """
        The reward depends on how closely the robot reachs the desired pose 
        and how long it holds this pose
        """
        np.roll(self.buffer, -1) # Roll towwards left
        self.buffer[-1] = -norm(goal-achieved_goal) # The first element becomes the  
                                                    # the last. Replace it with 
                                                    # the latest sample 
        return self.buffer.mean()
