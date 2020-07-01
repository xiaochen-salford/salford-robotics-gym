import os
import pybullet as bullet
import pybullet_data

from collections import namedtuple
from attrdict import AttrDict
import functools

# explicitly deal with mimic joints
def control_gripper(robot, parent, children, mul, **kwargs):
    control_mode = kwargs.pop("controlMode")
    if control_mode == bullet.POSITION_CONTROL:
        pose = kwargs.pop("targetPosition")
        # move parent joint
        bullet.setJointMotorControl2(
                robot, parent.id, control_mode, target_position=pose, 
                force=parent.maxForce, maxVelocity=parent.maxVelocity)

        # move child joints
        for name in children:
            child = children[name]
            child_pose = pose * mul[child.name]
            bullet.setJointMotorControl2(
                    robot, child.id, control_mode, targetPosition=child_pose, 
                    force=child.maxForce, maxVelocity=child.maxVelocity) 

    else:
        raise NotImplementedError("controlGripper does not support \"{}\" control mode".format(control_mode))
    # check if there 
    if len(kwargs) is not 0:
        raise KeyError("No keys {} in controlGripper".format(", ".join(kwargs.keys())))


def setup_sisbot(bullet, robot):
    control_joints = ["shoulder_pan_joint","shoulder_lift_joint", 
                        "elbow_joint", "wrist_1_joint", 
                        "wrist_2_joint", "wrist_3_joint", 
                        "robotiq_85_left_knuckle_joint"]
    joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    joint_num = bullet.getNumJoints(robot)
    joint_info = namedtuple( 
            "jointInfo", 
            ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity","controllable"])
    joints = AttrDict()
    for i in range(joint_num):
        info = bullet.getJointInfo(robot, i)
        joint_id = info[0]
        joint_name = info[1].decode("utf-8")
        joint_type = joint_type_list[info[2]]
        joint_lower_limit = info[8]
        joint_upper_limit = info[9]
        joint_max_force = info[10]
        joint_max_velocity = info[11]
        controllable = True if joint_name in control_joints else False
        info = joint_info( 
                joint_id, joint_name, joint_type, 
                joint_lower_limit, joint_upper_limit, 
                joint_max_force, joint_max_velocity,controllable
        )
        if info.type=="REVOLUTE": # set revolute joint to static
            bullet.setJointMotorControl2(robot, info.id, bullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            joints[info.name] = info

    mimic_parent_name = "robotiq_85_left_knuckle_joint"
    mimic_children = {"robotiq_85_right_knuckle_joint": 1, 
                      "robotiq_85_right_finger_joint": 1, 
                      "robotiq_85_left_inner_knuckle_joint": 1, 
                      "robotiq_85_left_finger_tip_joint": 1, 
                      "robotiq_85_right_inner_knuckle_joint": 1, 
                      "robotiq_85_right_finger_tip_joint": 1}
    parent = joints[mimic_parent_name] 
    children = AttrDict((j, joints[j]) for j in joints if j in mimic_children.keys())
    control_robotiq_c2 = functools.partial(control_gripper, robot, parent, children, mimic_children)

    return joints, control_robotiq_c2, control_joints, mimic_parent_name


phyx_client = bullet.connect(bullet.GUI)
bullet.setAdditionalSearchPath(pybullet_data.getDataPath())

curr_dir = os.path.dirname(os.path.realpath(__file__))
sisbot_model_path = curr_dir+"/../models/mdl_ur5robotiqc2.urdf"
plane_model_path = curr_dir+"/../models/plane/plane.urdf"

# setup sisbot
robot_start_pos = [0,0,0]
robot_start_orn =bullet.getQuaternionFromEuler([0,0,0])
print("----------------------------------------")
print("Loading robot from {}".format(sisbot_model_path))
robot = bullet.loadURDF(sisbot_model_path, robot_start_pos, robot_start_orn, flags=bullet.URDF_USE_INERTIA_FROM_FILE)
plane = bullet.loadURDF(plane_model_path)
joints, control_robotiq_c2, control_joints, mimic_parent_name = setup_sisbot(bullet, robot)
eef_id = 7 # ee_link

# start simulation
ABSE = lambda a,b: abs(a-b)
try:
    flag = True
    x_in = bullet.addUserDebugParameter("x", -2, 2, 0)
    y_in = bullet.addUserDebugParameter("y", -2, 2, 0)
    z_in = bullet.addUserDebugParameter("z", 0.5, 2, 1)
    i = 0
    while flag:
        i += 1
        print ("i : ", i) 
        x = bullet.readUserDebugParameter(x_in)
        y = bullet.readUserDebugParameter(y_in)
        z = bullet.readUserDebugParameter(z_in)

        joint_pose = bullet.calculateInverseKinematics(robot, eef_id, [x,y,z])
        for i, name in enumerate(control_joints):
            joint = joints[name]
            pose = joint_pose[i]
            rXYZ = bullet.getLinkState(robot, eef_id)[0] # real XYZ
            joint_states = bullet.getJointStates(robot, range(3))
            print("x_err= {:.2f}, y_err= {:.2f}, z_err= {:.2f}".format(*list(map(ABSE,[x,y,z],rXYZ))))

        bullet.stepSimulation()
    bullet.disconnect()
except:
    bullet.disconnect()

