from srg.envs.bullet3.gyms import PoseEnv
import pybullet as bullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np
import pdb

pi = 3.14159267

class RandomAgent:
    """the world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation=None, reward=None, done=None):
        return self.action_space.sample()

if __name__ == '__main__':
    env = PoseEnv()
    agent = RandomAgent(env.robot.action_space)
    goal = np.array([0.5, 0.5, 0.5])
    done = False
    flag = True
    num = 10000
    try:
        while True:
            action = {}
            # action['arm'] = [100, 100, 100, 100, 100, 100]
            # action['arm'] = [250, 0, 0, 0, 0, 0]
            # action['arm'] = [0, 100, 0, 0, 0, 0]
            # action['arm'] = [0, 0, 0, 500, 0, 0]
            # action['arm'] = [0, 0, 0, 0, 400, 0]
            # action['arm'] = [pi/4, 0, 0, 0, 0, 0]
            # action['arm'] = [0, 0, pi/4, 0, 0, pi/3]
            action = np.array([0, 0, pi/4, 0, 0, pi/3])

            next_state, reward, done, _ = env.step(action, goal)
            num -= 1
            if num == 0:
                num = 10000
                env.reset()

            # pdb.set_trace()    
        bullet.disconnect()
    except:
        bullet.disconnect()


