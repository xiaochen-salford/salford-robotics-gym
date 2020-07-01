from srg.envs.bullet3.fetch_gym import GraspEnv
import pybullet as bullet
import numpy as np


def main():
    env = GraspEnv()
    action_dim = env.robot.dof_num 
    zero_action = np.zeros(action_dim)
    try:
        while True:
            env.step(zero_action)

        bullet.disconnect()
    except:
        bullet.disconnect()


if __name__ == '__main__':
    main()


