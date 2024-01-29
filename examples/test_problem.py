#!/usr/bin/env python

from __future__ import print_function

import pybullet as p
import time
import random
from itertools import islice, count
import numpy as np
from pybullet_tools.pr2_primitives import Pose, Conf, get_ik_ir_gen, get_motion_gen, \
    get_stable_gen, Attach, Detach, Clean, Cook, control_commands, Grasp, \
    get_gripper_joints, GripperCommand,get_ik_fn, apply_commands, State
from pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, \
    get_free_motion_gen, get_holding_motion_gen
from pybullet_tools.pr2_utils import TOP_HOLDING_LEFT_ARM, PR2_URDF, PR2_TOOL_FRAMES,ARM_NAMES, DRAKE_PR2_URDF, \
    SIDE_HOLDING_LEFT_ARM,get_base_pose,get_side_grasps,compute_grasp_width, get_top_grasps,PR2_GROUPS, open_arm, get_disabled_collisions, REST_LEFT_ARM, rightarm_from_leftarm
from pybullet_tools.utils import set_base_values, joint_from_name, quat_from_euler, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, wait_if_gui, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose,BLOCK_URDF, wait_if_gui, load_pybullet, set_quat, Euler, PI, RED, add_line, \
    wait_for_duration, LockRenderer, base_aligned_z, Point, set_point, get_aabb, stable_z_on_aabb, AABB, \
    WorldSaver, enable_gravity,GraspInfo, dump_world, set_pose, get_unit_vector, \
    draw_global_system, draw_pose, set_camera_pose, Pose, Point, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, wait_if_gui, disconnect, DRAKE_IIWA_URDF, wait_if_gui, update_state, disable_real_time, HideOutput, \
    get_model_path, draw_pose, get_max_limit, get_movable_joints, set_joint_position, unit_pose, create_box, RED, set_point, \
    unit_quat, stable_z, set_camera_pose, LockRenderer, add_line, multiply, invert, get_relative_pose, GREEN, BLUE, TAN, create_cylinder


BASE_EXTENT = 3.5 # 2.5
BASE_LIMITS = (-BASE_EXTENT*np.ones(2), BASE_EXTENT*np.ones(2))
GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
SELF_COLLISIONS = False

LEFT_ARM = 'left'
arm = 'left'
def get_grasp_gen(robot, collisions=False, randomize=True):
    tool_link = link_from_name(robot, PR2_TOOL_FRAMES[LEFT_ARM])
    def gen(body):
        grasps = []
        approach_vector = APPROACH_DISTANCE*get_unit_vector([1, 0, 0])
        grasps.extend(Grasp('top', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM)
                        for g in get_top_grasps(body, grasp_length=GRASP_LENGTH))
        
        #approach_vector = APPROACH_DISTANCE*get_unit_vector([2, 0, -1])
        #grasps.extend(Grasp('side', body, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_LEFT_ARM)
        #                for g in get_side_grasps(body, grasp_length=GRASP_LENGTH))
    
        filtered_grasps = []
        for grasp in grasps:
            grasp_width = compute_grasp_width(robot, arm, body, grasp.value) if collisions else 0.0
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)
        if randomize:
            random.shuffle(filtered_grasps)
        return [(g,) for g in filtered_grasps]
    return gen

def plan(robot, block, fixed, teleport):
    grasp_gen = get_grasp_gen(robot, 'top')
    ik_fn = get_ik_fn(robot, fixed=fixed, teleport=teleport)
    free_motion_fn = get_motion_gen(robot, fixed=([block] + fixed), teleport=teleport)
    holding_motion_fn = get_holding_motion_gen(robot, fixed=fixed, teleport=teleport)

    pose0 = BodyPose(block)
    conf0 = BodyConf(robot)
    saved_world = WorldSaver()
    #listofgrasps = list(grasp_gen(block))
    #problems move table
    g = multiply(get_gripper_pose(robot), (np.array([0.05, 0.05, 0.05]), unit_quat())) # get_pose of the end-effecter +  [0.05, 0.05, 0.05]
    approach_vector = APPROACH_DISTANCE*get_unit_vector([1, 0, 0])
    listofgrasps = [Grasp('top', block, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM)]
    print(len(listofgrasps))
    for grasp, in listofgrasps:
        saved_world.restore()
        result1 = ik_fn(arm, block, pose0, grasp, conf0)
        if result1 is None:
            print(result1)
            continue
        conf1, path2 = result1
        print(conf1)
        #pose0.assign()
        #result2 = free_motion_fn(conf0, conf1)
        #if result2 is None:
        #    continue
        #path1, = result2
        #result3 = holding_motion_fn(conf1, conf0, block, grasp)
        #if result3 is None:
        #    continue
        #path3, = result3
        return Command(conf1.savers)
    return None


def main(display='execute'): # control | execute | step
    connect(use_gui=True)
    disable_real_time()
    draw_global_system()
    with HideOutput():
        pr2_urdf = DRAKE_PR2_URDF
        robot = load_pybullet(pr2_urdf, fixed_base=True)# KUKA_IIWA_URDF | DRAKE_IIWA_URDF
        #floor = load_model('models/short_floor.urdf')
        table_path = "models/table_collision/table.urdf"
        table = load_pybullet(table_path, fixed_base=True)
        set_pose(table, Pose(Point(x=0.5, z = -0.25)))
        set_quat(table, quat_from_euler(Euler(yaw=PI/2)))
    block = load_model(BLOCK_URDF, fixed_base=False)
    set_pose(block, Pose(Point(x=0.75, z=stable_z(block, table))))
    set_default_camera(distance=3)
    dump_world()

    saved_world = WorldSaver()
    command = plan(robot, block, fixed=[table], teleport=False)
    if (command is None) or (display is None):
        print('Unable to find a plan!')
        return

    saved_world.restore()
    update_state()
    wait_if_gui('{}?'.format(display))
    if display == 'control':
        enable_gravity()
        command.control(real_time=False, dt=0)
    elif display == 'execute':
        command.refine(num_steps=10).execute(time_step=0.005)
    elif display == 'step':
        command.step()
    else:
        raise ValueError(display)

    print('Quit?')
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()