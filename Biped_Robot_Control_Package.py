# Author: Van Thanh Nguyen - HUR group's Ph.D. student - GIST
import numpy as np

class PD_Controller: # MIMO, joint position PD control
    def __init__(self, Kp, Kd, n, sampling_time):
        self.n = n
        self.error_pre = np.zeros((n, 1))
        self.Kp = Kp
        self.Kd = Kd
        self.sampling_time = sampling_time # in sec
        
    def PD_Control_Calculate (self, ref_val, fb_val): # MIMO
        error = ref_val - fb_val
        error_d = (error - self.error_pre)/self.sampling_time
        output = self.Kp@error + self.Kd@error_d
        self.error_pre = error
        return output # return a vector of control output
    
    def Reset (self):
        self.error_pre = np.zeros(1,self.n)

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

class Biped_Robot_Control_Sytem:
    def __init__(self, mj_data, JointPosPDController):

        self.mj_data = mj_data
        self.Joint_PDController = JointPosPDController # init joint's PD position controller
        self.L1 = 0.3395 # m, thigh's kinematic length
        self.L2 = 0.321695 # m, shank's kinematic length
        self.L3 = 0 # offset from the ankle to the foot surface
        self.hip_width = 0.170807*2 # distance between 2 hip join points
        self.hip_pitch_delta_angle = np.deg2rad(15) # offset of the hip's pitch joint
        self.g = 9.81 # m/s2, gravity const
        
        self.Right_Hip_Com_Offset = np.transpose(np.array([0, 0, -self.hip_width/2, 1])) # -0.128278 offset from pelvis center to hip joint
        self.Left_Hip_Com_Offset = np.transpose(np.array([0, 0, self.hip_width/2, 1]))
        self.CoM_height_offset  = 0.0 # adjust this to estimate CoM
        self.Zc = 0.6 + self.CoM_height_offset
        self.w = np.sqrt(self.g/self.Zc)
        self.dt = 0.001 #self.env.model.opt.timestep # time step in mujoco simu

        self.left_ankle_pos = np.array([0, self.hip_width/2, 0, 1]) # init ankle pos, set ref frame in the middle
        self.right_ankle_pos = np.array([0, -self.hip_width/2, 0, 1])
        
    def step_ref (self, T, ref_angl): # to generate referenc angle by ramping it gradually
        t = self.mj_data.time
        if  t <= T:
            angle = (6*(t/T)**5 - 15*(t/T)**4 + 10*(t/T)**3)*ref_angl
            return angle
        else:
            return ref_angl
        
    def Init_Position (self, R_tform, L_tform, T):
        # R_tform and L_tform represent initial target posture of the robot
        # T is the ramping time for ref's transition
        thetaR_des = self.InvBody2Foot_Revise(R_tform) # calculate inverse kinematic to get reference target joint angle
        thetaL_des = self.InvBody2Foot_Revise(L_tform)
        
        # normal hip's reference angle
        R_theta_ref = self.step_ref(T, thetaR_des)
        L_theta_ref = self.step_ref(T, thetaL_des)       

        # pack the reference angle
        robot_joint_ref_angle = np.array([R_theta_ref,L_theta_ref]).reshape(-1,1) # reshape to 1-d vector
        return robot_joint_ref_angle

    def InvBody2Foot_Revise(self, tform): # This is the Inverse Kinematic, calculate joints' angles from desired position and orientation

        L3 = self.L1   # m, Upper leg length
        L4 = self.L2 # m, Lower leg length
        L5 = self.L3     # m, Ankle to foot contact offset

        # Extract position/orientation information
        R = tform[:3,:3]
        p = tform[:3, 3]
        # p = np.transpose(np.array([p[2], -p[1], p[0]]))
        # 2) Get inverse rotation matrix (in this case, a transpose)
        Rp = np.transpose(R)
        n = Rp[:,0]
        s = Rp[:,1]
        a = Rp[:,2]
        p = -Rp@p # position of hip joint w.r.t ankle/foot's frame
        
        # 3) Compute analytic solution
        cos4 = ((p[0]+L5)**2 + p[1]**2 + p[2]**2 - L3**2 - L4**2)/(2*L3*L4)
        temp = 1 - cos4**2
        if temp < 0:
            temp = 0
            print('Waning: Unable to reach desired end-effector position/orientation')
            return
        th4 = np.atan2(np.sqrt(temp),cos4) # Knee joint

        # note: you can put -sqrt(temp) to change direction of knee bending
        temp = (p[0]+L5)**2+p[1]**2
        if temp < 0:
            temp = 0
            print('Warning: Unable to reach desired end-effector position/orientation')
            return
        #r = np.sqrt(p[0]**2+p[1]**2+p[2]**2)
        th5 = np.atan2(-p[2],np.sqrt(temp))-np.atan2(np.sin(th4)*L3,np.cos(th4)*L3+L4) # ankle pitch
        th6 = np.atan2(p[1],-p[0]-L5) # ankle roll

        # Calculate theta1, 2 and 3:
        a11 = np.cos(th4+th5)
        a12 = -np.sin(th4+th5)
        a21 = np.sin(th4+th5)
        a22 = np.cos(th4+th5)
        
        b1 = np.cos(th6)*a[0]-np.sin(th6)*a[1]
        b2 = a[2]

        A = np.array([[a11, a12],
                    [a21, a22]])
        B = np.transpose(np.array([b1, b2]))

        if np.linalg.det(A) == 0:
            print("out of range")
            return
        temp = np.linalg.inv(A)@B

        th2 = np.arcsin(temp[0])
        C2 = np.cos(th2)
        #S3 = a12/C2
        C3 = (np.sin(th6)*a[0]+np.cos(th6)*a[1])/C2
        th3 = np.arccos(C3)
        #th3 = np.atan2(S3,C3)

        a11 = -np.sin(th2)*np.sin(th3)*np.sin(th4+th5)-np.cos(th2)*np.cos(th4+th5)
        a12 = -np.cos(th3)*np.sin(th4+th5)
        a21 = np.sin(th2)*np.sin(th3)*np.cos(th4+th5) - np.sin(th4+th5)*np.cos(th2)
        a22 = np.cos(th3)*np.cos(th4+th5)
        
        b1 = np.cos(th6)*s[0] - np.sin(th6)*s[1]
        b2 = s[2]

        A = np.array([[a11, a12],
                    [a21, a22]])
        B = np.transpose(np.array([b1, b2]))
        temp = np.linalg.inv(A)@B
        th1 = np.atan2(temp[0], temp[1])

        th = np.array([th1, th2, th3, th4, th5, th6]) # hip flex, hip abd, hip int, knee, ankle pitch, ankle roll
        # Convert to normal config
        th = self.Normal_Config_Joint_Convert(th)
        return th
    
    def Hip_Joint_Convert (self, theta1_ref, theta2_ref, theta3_ref):
        # this function transforms the hip joint from the normal kinematic configuration to the
        # offset configuration at the hip pitch joint
        # Input:
        # theta1_ref, theta2_ref, theta3_ref: nornal hip joint's angle (refer to the normal configuaraion)
        # L1: kinematic length of the thigh        
        # delta: offset angle of the hip pitch joint
        # return the equivalent hip joint w.r.t the offset configuration

        d1 = 0 # d1: offset distance from the pelvis center to the hip's joint' point
        L1 = self.L1
        delta = self.hip_pitch_delta_angle

        a11 = -L1*np.cos(delta)
        a12 = L1*np.sin(delta)
        a21 = -L1*np.sin(delta)
        a22 = -L1*np.cos(delta)

        b1 = L1*np.sin(theta2_ref) - d1 + d1*np.cos(delta)
        b2 = -L1*np.cos(theta1_ref)*np.cos(theta2_ref) + d1*np.sin(delta)

        A = np.array([[a11, a12], [a21, a22]])
        B = np.array([[b1], [b2]])

        temp = np.linalg.inv(A)@B

        theta2 = delta - np.arcsin(temp[0])
        theta2 = theta2[0]

        c1_cdelta_2 = temp[1]
        s1_cdelta_2 = np.sin(theta1_ref)*np.cos(theta2_ref)

        theta1 = np.atan2(s1_cdelta_2,c1_cdelta_2)
        theta1 = theta1[0]

        c11 = np.sin(theta1)*np.sin(delta-theta2)
        c12 = np.cos(theta1)
        c21 = np.sin(delta)*np.sin(delta-theta2)*np.cos(theta1) +np.cos(delta)*np.cos(delta-theta2)
        c22 = -np.sin(delta)*np.sin(theta1)
    
        d1 = -np.sin(theta1_ref)*np.sin(theta2_ref)*np.sin(theta3_ref) + np.cos(theta1_ref)*np.cos(theta3_ref)
        d2 = np.sin(theta3_ref)*np.cos(theta2_ref)

        C = np.array([[c11, c12], [c21, c22]])
        D = np.array([[d1], [d2]])

        temp1 = np.linalg.inv(C)@D
        theta3 = np.atan2(temp1[0], temp1[1])
        theta3 = theta3[0] # to get a single number

        return [theta1, theta2, theta3]
    
    def Normal_Config_Joint_Convert (self, theta_ref):
        th_ref = theta_ref

        th1_ref, th2_ref, th3_ref = self.Hip_Joint_Convert(theta_ref[0], theta_ref[1], theta_ref[2])

        th_ref[0] = th1_ref
        th_ref[1] = th2_ref
        th_ref[2] = th3_ref

        return th_ref