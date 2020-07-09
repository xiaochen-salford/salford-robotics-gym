import os
import pybullet as bullet

class RobotEnv:
    """ 
    The base class for all robots 
    RobotEnv is sub-env for GymEnv; it is used to specify a robot and sould be attached to RobotEnv
    """
    def __init__(self, config, initial_pos, inital_orn):
        model_path = config.model_path
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'models', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not found'.format(fullpath))

        self.id = bullet.loadURDF(fullpath, initial_pos, inital_orn, flags=bullet.URDF_USE_INERTIA_FROM_FILE)
        for idx in range(config.joints_num):
            bullet.enableJointForceTorqueSensor(self.id, idx, enableSensor=True)

        self.action_space = self.init_action_space()
        self.state_space = self.init_state_space()
      
    def init_action_space(self):
        raise NotImplementedError
    
    def init_state_space(self):
        raise NotImplementedError

    def set_action(self, action, **kwargs):
        raise NotImplementedError
    
    def get_state(self):
        raise NotImplementedError

    def get_action_size(self):
        raise NotImplementedError

    def get_state_size(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

