import pybullet as p
import time
import os 
curr_dir = os.path.dirname(os.path.realpath(__file__))

p.connect(p.GUI)
p.loadURDF(curr_dir+"/../models/plane/plane.urdf")
robot = p.loadURDF(curr_dir+"/../models/cassie/urdf/cassie_collide.urdf",[0,0,0.8], useFixedBase=False)
gravId = p.addUserDebugParameter("gravity",-10,10,-10)
jointIds=[]
paramIds=[]

p.setPhysicsEngineParameter(numSolverIterations=100)
p.changeDynamics(robot,-1,linearDamping=0, angularDamping=0)

jointAngles=[0,0,1.0204,-1.97,-0.084,2.06,-1.9,0,0,1.0204,-1.97,-0.084,2.06,-1.9,0]
activeJoint=0
for j in range (p.getNumJoints(robot)):
	p.changeDynamics(robot,j,linearDamping=0, angularDamping=0)
	info = p.getJointInfo(robot,j)
	#print(info)
	jointName = info[1]
	jointType = info[2]
	if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
		jointIds.append(j)
		paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"),-4,4,jointAngles[activeJoint]))
		p.resetJointState(robot, j, jointAngles[activeJoint])
		activeJoint+=1

p.setRealTimeSimulation(1)

while(1):
	p.getCameraImage(320,200)
	p.setGravity(0,0,p.readUserDebugParameter(gravId))
	for i in range(len(paramIds)):
		c = paramIds[i]
		targetPos = p.readUserDebugParameter(c)
		p.setJointMotorControl2(robot,jointIds[i],p.POSITION_CONTROL,targetPos, force=140.)	
	time.sleep(0.01)
