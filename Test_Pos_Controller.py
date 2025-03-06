import mujoco 
import numpy as np
from mujoco.viewer import launch_passive # to handle the mujoco GUI

import Biped_Robot_Control_Package as Ctrl_System # user's library to include PD controller and other useful functions
import json # to save the data


def Get_Joints_Pos (mj_model, mj_data): # To get all robot's joint position
    q = []
    joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
    for joint in joint_names:
        q.append(mj_data.joint(joint).qpos)
    n = len(q)
    if n> 12:
        q = q[1:] # ignore the position and orientation of the torso, only take the joint position

    #order for each 6dof of 1 leg: hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll
    # first 6 values belong to right leg, and last 6 values belong left leg
    q = np.array([q[0], q[1], q[2], q[3], q[5], q[4], q[6], q[7], q[8], q[9], q[11], q[10]])
    return q  

save_data = [] # just to save the simulation data

if __name__ == "__main__":


    nf_ref = np.array([-1, 0, 0, 0]) # init orientation matrix
    sf_ref = np.array([0, 0, 1, 0]) # set like this to make the foot parallel to the ground
    af_ref = np.array([0, 1, 0, 0]) #

    # -0.6 is the position in x-axis, need to refer to DH coordinate of the leg, set to
    # set to -0.6 to bend the knee
    R_pf_ref = np.array([-0.6, 0, 0, 1]) # right leg, position vector representing ankle position w.r.t hip joint
    L_pf_ref = np.array([-0.6, 0, 0, 1]) # left leg, position vector representing ankle position w.r.t hip joint


    R_tform = np.transpose(np.array([nf_ref, sf_ref, af_ref, R_pf_ref])) # right leg, assemble to homogeneous transformation matrix
    L_tform = np.transpose(np.array([nf_ref, sf_ref, af_ref, L_pf_ref])) # left leg, assemble to homogeneous transformation matrix


    # Load .xml model
    xml_file_robot_h1 = 'mujoco_model/Bipedal_Robot.xml' # the string represent the direction to the .xml file
    mj_model = mujoco.MjModel.from_xml_path(xml_file_robot_h1) # creat mujoco model
    mj_data = mujoco.MjData(mj_model) # creat mujoco data, mujoco data stores all the states like joint's position and velocity, time...
    time_step = mj_model.opt.timestep # get time step in the simulation, here is 0.001s

    
    # optional, visualize contact frames and forces, make body transparent
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # scale force vector of contact visualization elements, to make it be good visible
    mj_model.vis.scale.contactwidth = 0.3
    mj_model.vis.scale.contactheight = 0.3
    mj_model.vis.scale.forcewidth = 0.2
    mj_model.vis.map.force = 0.01

    # 1st 6 elements correspond to Right leg, the rest 6 ones correspond to Left Leg
    # Set PD controller' gains
    Joint_Kp = np.diag([500,300,300,500,200, 200,   500, 300, 300, 500, 200, 200]) # N.m/rad
    Joint_Kd = np.diag([6.25, 3.25, 3.25, 6.25, 1.25, 1.25,   6.25, 3.25, 3.25, 6.25, 1.25, 1.25]) #N.ms/rad
    n_actuated = Joint_Kp.shape[0] # number of actuators, 12

    PDController= Ctrl_System.PD_Controller(Joint_Kp, Joint_Kd, n_actuated, time_step) # init robot's joint PD controller
    Robot_Control_System = Ctrl_System.Biped_Robot_Control_Sytem(mj_data, PDController) # init robot's control system


    T = 0.8 # sec, init time at the beginning

    viewer = launch_passive(mj_model, mj_data) # create a passive viewer to handle the GUI

    #while viewer.is_running(): # optional, if we do not want to stop the simulation
    while mj_data.time <= 3: # optional,if we want to run the simulation in a certain time

        mujoco.mj_step(mj_model, mj_data) # calculate forward dynamics for each time step
        simu_time = mj_data.time # get simulation time

        # the Init_Position() function basically bends the knee joint of the robot.
        # it solve the inverse kinematic to get robot's joint reference angle
        robot_joint_ref_angle = Robot_Control_System.Init_Position(R_tform, L_tform, T) # T is the moving time in the ramping function of the joint's trajectory

        # Calculate joints' torques
        # Joint Position Control Layer: Use conventional PD controller
        robot_joint_feb_angl = Get_Joints_Pos(mj_model, mj_data) # Get feedback position
        Joint_Tqr = Robot_Control_System.Joint_PDController.PD_Control_Calculate(robot_joint_ref_angle, robot_joint_feb_angl) # calculate torque by PD controller

        # set calculated joints' torques to the robot
        mj_data.ctrl = np.array(Joint_Tqr).reshape(1,-1) # apply joint torque in mujoco, reshape to row vector to match the size         

        # save simulation data
        save_data.append([mj_data.time, robot_joint_feb_angl[0].item(), robot_joint_feb_angl[1].item(), robot_joint_feb_angl[2].item()
                          ,robot_joint_feb_angl[3].item(),robot_joint_feb_angl[4].item(),robot_joint_feb_angl[5].item()])
        
        # enable contact force and points visualizaiton
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        viewer.sync() # update the GUI after calculing forward dynamic

    viewer.close() # stop simu

    # save the simulation data to file
    # use plot_data.py to plot
    with open('save_data.json', 'w') as file:
        json.dump(save_data, file)