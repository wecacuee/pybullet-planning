import os
import math 
import numpy as np
import time
import pybullet as p
import random
from datetime import datetime
import pybullet_data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from collections import namedtuple
from collections import deque
from attrdict import AttrDict


ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
CUBE_URDF_PATH = "./ur_e_description/urdf/cube_red.urdf"


# Actions
ACTION_MOVE_TO_OBJECT = 0
ACTION_PICK_UP = 1
ACTION_MOVE_TO_PLACE = 2
ACTION_PLACE = 3

# NUMBER OF ACTIONS
num_actions = 4


#Q-learning parameters
state_space_dim = 21
learning_rate = 0.1
discount_factor = 0.99
num_episodes = 1000
max_steps_per_episode = 100
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.01


#def objDistance(a1, a2):
 #   (x, y, z) = a1[0], a1[1] , a1[2] 
  #  (x2, y2, z2) = a2[0], a2[1] , a2[2] 
   # return math.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)    
def objDistance(obj1, obj2):
    (x, y, z) =  obj1[0], obj1[1] , obj1[2]
    (x2, y2, z2) = obj2[0], obj2[1] , obj2[2]
    return math.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2) 
  
class UR5Sim():
  
    def __init__(self, camera_attached=False):
        p.connect(p.GUI)
        p.setRealTimeSimulation(True)
        
        self.end_effector_index = 7
        self.ur5 = self.load_robot()
        self.num_joints = p.getNumJoints(self.ur5)
        self.cube = self.load_cube()
        
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = p.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                p.setJointMotorControl2(self.ur5, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info     


    def load_robot(self):
        flags = p.URDF_USE_SELF_COLLISION
        table = p.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])
        robot = p.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        return robot
    
    def load_cube(self):
        cube = p.loadURDF(CUBE_URDF_PATH, [0.4, 0.0, 0.5], [0, 0, 0, 1])
        return cube
        
    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        p.setJointMotorControlArray(
            self.ur5, indexes,
            p.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.04]*len(poses), forces=forces
        )


    def get_joint_angles(self):
        j = p.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints
    

    def check_collisions(self):
        collisions = p.getContactPoints()
        if len(collisions) > 0:
            print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False


    def calculate_ik(self, position, orientation):
        quaternion = p.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]

        joint_angles = p.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )
        return joint_angles
       

    def add_gui_sliders(self):
        self.sliders = []
        self.sliders.append(p.addUserDebugParameter("X", 0, 1, 0.4))
        self.sliders.append(p.addUserDebugParameter("Y", -1, 1, 0))
        self.sliders.append(p.addUserDebugParameter("Z", 0.3, 1, 0.4))
        self.sliders.append(p.addUserDebugParameter("Rx", -math.pi/2, math.pi/2, 0))
        self.sliders.append(p.addUserDebugParameter("Ry", -math.pi/2, math.pi/2, 0))
        self.sliders.append(p.addUserDebugParameter("Rz", -math.pi/2, math.pi/2, 0))


    def read_gui_sliders(self):
        x = p.readUserDebugParameter(self.sliders[0])
        y = p.readUserDebugParameter(self.sliders[1])
        z = p.readUserDebugParameter(self.sliders[2])
        Rx = p.readUserDebugParameter(self.sliders[3])
        Ry = p.readUserDebugParameter(self.sliders[4])
        Rz = p.readUserDebugParameter(self.sliders[5])
        return [x, y, z, Rx, Ry, Rz]
        
    def get_current_pose(self):
        linkstate = p.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)
        
    def get_object_pose(self):
        # Get the current position and orientation of the cube
        linkstate = p.getBasePositionAndOrientation(self.cube)
        position, orientation = linkstate[0], linkstate[1]
        return position, orientation


	   
    def get_state(self):
        robot_joint_angles = self.get_joint_angles()
        robot_position, robot_orientation = self.get_current_pose()
        object_position, object_orientation = self.get_object_pose()

        # Determine the stage based on the distance between the robot and the object
        stage = 0 if objDistance(robot_position, object_position) > 0.07 else 1
        robot_orientation_euler = p.getEulerFromQuaternion(robot_orientation)
        object_orientation_euler = p.getEulerFromQuaternion(object_orientation)
	
        # Check if the object is picked up or not 
        is_picked_up = False
        is_done = False
        state = [
            stage,
            robot_joint_angles[0],    	# Joint 1 angle
            robot_joint_angles[1],    	# Joint 2 angle
            robot_joint_angles[2],    	# Joint 3 angle
            robot_joint_angles[3],    	# Joint 4 angle
            robot_joint_angles[4],    	# Joint 5 angle
            robot_joint_angles[5],    	# Joint 6 angle
            robot_position[0],        	# End-effector x position
            robot_position[1],        	# End-effector y position
            robot_position[2],          # End-effector z position
            robot_orientation_euler[0], # End-effector roll orientation
            robot_orientation_euler[1], # End-effector pitch orientation
            robot_orientation_euler[2], # End-effector yaw orientation
            object_position[0],       	# Object x position
            object_position[1],       	# Object y position
            object_position[2],       	# Object z position
            object_orientation_euler[0], # End-effector roll orientation
            object_orientation_euler[1], # End-effector pitch orientation
            object_orientation_euler[2], # End-effector yaw orientation
            is_done,
            is_picked_up              # Flag indicating if the object is picked up or not
        ]

        return state
    def reset(self):
        # Reset the robot's position to the initial position
        initial_joint_angles = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]
        self.set_joint_angles(initial_joint_angles)

        # Reset the object's position to a random position within a specified range
        min_x, max_x = 0.35, 0.45
        min_y, max_y = -0.05, 0.05
        min_z, max_z = 0.48, 0.52
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        z = random.uniform(min_z, max_z)
        p.resetBasePositionAndOrientation(self.cube, [x, y, z], [0, 0, 0, 1])

    
    def perform_action(self, action):
        if action == ACTION_MOVE_TO_OBJECT:
            object_position, object_orientation = self.get_object_pose()
            desired_end_effector_position = object_position
            desired_end_effector_orientation = p.getEulerFromQuaternion(object_orientation)

            joint_angles = self.calculate_ik(desired_end_effector_position, desired_end_effector_orientation)
            self.set_joint_angles(joint_angles)
            
            state = self.get_state()
            collision = self.check_collisions()
            reward = calculate_reward(state, action)
            done = False
            return state, reward, done 
        elif action == ACTION_PICK_UP:
            # Pick up the object
            # Update the state 
            # Calculate reward 
            # Return the new state, reward, and done
            pass

        elif action == ACTION_MOVE_TO_PLACE:
            # Move the gripper to the place
            # Update the state 
            # Calculate reward based on the action and the updated state
            # Return the new state, reward, and 'done' flag
            pass

        elif action == ACTION_PLACE:
            # Place the object
            # Update the state 
            # Calculate reward based on the action and the updated state
            # Return the new state, reward, and 'done' flag
            pass

def calculate_reward(current_state, action):
        stage = current_state[0]
        robot_gripper_pos = current_state[7:10]
        object_pos = current_state[13:16]
        # If object is not in the gripper
        if stage == 0:
            reward =  -objDistance(robot_gripper_pos, object_pos)
        else:
            destination = [dest_x,dest_y, dest_z] # destination coordinates
            reward = -objDistance(robot_gripper_pos, destination)   
        return reward
        


#robot_sim = UR5Sim()
q_table = np.zeros((state_space_dim, num_actions))

def demo_simulation():
    """ Demo program showing how to use the sim """
    robot_sim = UR5Sim()
    robot_sim.add_gui_sliders()
    while True:
        x, y, z, Rx, Ry, Rz = robot_sim.read_gui_sliders()
        joint_angles = robot_sim.calculate_ik([x, y, z], [Rx, Ry, Rz])
        robot_sim.set_joint_angles(joint_angles)
        robot_sim.check_collisions()
        q_table = np.zeros((state_space_dim, num_actions))
        exploration_rate = max_exploration_rate
        for episode in range(num_episodes):
            robot_sim.reset()  # Reset the robot's position and object's position for each episode
            state = robot_sim.get_state()  # Get the initial state of the environment
            total_reward = 0

            for step in range(max_steps_per_episode):
                # Exploration-exploitation trade-off (epsilon-greedy)
                exploration_threshold = random.uniform(0, 1)
                if exploration_threshold  > exploration_rate:
                    action = np.argmax(q_table[state, :])
                else:
                    action =  0
                    #random.randint(0, num_actions - 1)

                # Execute the action and observe the new state and reward
                new_state, reward, done = robot_sim.perform_action(action)

                # Q-value update using the Q-learning formula
                q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])

                state = new_state
                total_reward += reward

                # Check if the episode is done (goal reached or max steps reached)
                if done:
                    break

            # Decay exploration rate for the next episode
            exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    

if __name__ == "__main__":
    demo_simulation()
    time.sleep(5)
