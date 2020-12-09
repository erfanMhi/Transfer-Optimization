
import numpy as np
from math import cos, sin, pi, sqrt
from copy import deepcopy

class Arm:
    def __init__(self, lengths, just_end_effctor_pos=True):
        self.n_dofs = len(lengths)
        self.lengths = np.concatenate(([0], lengths))
        self.joint_xy = []
        self.just_end_effctor_pos = just_end_effctor_pos

    def fw_kinematics(self, p):
        assert(len(p) == self.n_dofs)
        p = np.append(p, 0)
        self.joint_xy = []
        end_effector_pos = None
        mat = np.identity(4)
        for i in range(0, self.n_dofs + 1):
            m = [[cos(p[i]), -sin(p[i]), 0, self.lengths[i]],
                 [sin(p[i]),  cos(p[i]), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            mat = np.dot(mat, np.array(m))
            if not self.just_end_effctor_pos:
                v = np.dot(mat, np.array([0, 0, 0, 1]))
                self.joint_xy += [v[:2]]
            if i == self.n_dofs:
                end_effector_pos = np.dot(mat, np.array([0, 0, 0, 1]))[:2]
        if self.just_end_effctor_pos:
            return end_effector_pos
        else: 
            return end_effector_pos, self.joint_xy


def get_arm(length, joint_num):
    return Arm(np.ones(joint_num) * length / joint_num)



def get_distance_func(arm, max_angle, target_pos, obj_type='relative'):
    assert 0<=max_angle<=1, 'max_angle is not in range (0, 1)'
    
    if obj_type=='relative' or obj_type=='limited' or obj_type=='negative_obj':
        angular_range = max_angle * pi * 2.
    else:
        angular_range = pi * 2.
    def get_neg_distance(angles, verbos=False):

        if verbos:
            print('Angles: ', angles)

        if obj_type=='limited':
            angles[0 > angles] = 0.
            angles[1 < angles] = 1.
        elif obj_type=='negative_obj':
            neg_angles_ind = angles<(0.0)
            one_pos_angles_ind = angles>(1.0)
            if any(np.logical_or(neg_angles_ind, one_pos_angles_ind)):
                return ((np.sum(np.abs(angles[neg_angles_ind])) + np.sum(angles[one_pos_angles_ind]-1))*-1) + -3
                
        command = (angles - 0.5) * angular_range
        
        if obj_type=='absolute':
            max_angle_pi = max_angle*pi 
            command[command > max_angle_pi] = max_angle_pi
            command[command < -max_angle_pi] = -max_angle_pi  

        
        ef = arm.fw_kinematics(command)
        if verbos:
            print(ef)
        
        return -np.linalg.norm(ef - target_pos)
    
    return get_neg_distance

if __name__ == "__main__":
    # 1-DOFs
    a = Arm([1])
    v,_ = a.fw_kinematics([0])
    np.testing.assert_almost_equal(v, [1, 0])
    v,_ = a.fw_kinematics([pi/2])
    np.testing.assert_almost_equal(v, [0, 1])

    # 2-DOFs
    a = Arm([1, 1])
    v,_ = a.fw_kinematics([0, 0])
    np.testing.assert_almost_equal(v, [2, 0])
    v,_ = a.fw_kinematics([pi/2, 0])
    np.testing.assert_almost_equal(v, [0, 2])
    v,_ = a.fw_kinematics([pi/2, pi/2])
    np.testing.assert_almost_equal(v, [-1, 1])
    v,x = a.fw_kinematics([pi/4, -pi/2])
    np.testing.assert_almost_equal(v, [sqrt(2), 0])

    # a 4-DOF square
    a = Arm([1, 1, 1,1])
    v,_ = a.fw_kinematics([pi/2, pi/2, pi/2, pi/2])
    np.testing.assert_almost_equal(v, [0, 0])