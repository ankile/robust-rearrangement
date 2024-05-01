import math
from typing import Dict, List

import numpy as np
import torch
import cvxpy as cp
from scipy.spatial.transform import Rotation as R

import furniture_bench.controllers.control_utils as C

import numpy as np
import scipy


def diffik_factory(real_robot=True, *args, **kwargs):
    if real_robot:
        import torchcontrol as toco

        base = toco.PolicyModule
    else:
        base = torch.nn.Module

    class DiffIKController(base):
        """Differential Inverse Kinematics Controller"""

        def __init__(self):
            """Initialize Differential Inverse Kinematics Controller.

            Args:
            """
            super().__init__()
            self.ee_pos_desired = None
            self.ee_quat_desired = None
            self.ee_pos_error = None
            self.ee_rot_error = None

            self.pos_scalar = 4.0
            self.rot_scalar = 9.0

            self.scale_errors = True
            self.use_qp = True
        
        def qp_fwd(self, joint_pos_current, ee_velocity_desired, jacobian):
            v_max = 2.7 
            G = np.vstack([
                np.eye(len(joint_pos_current)),
                -1.0 * np.eye(len(joint_pos_current))])
            h = np.vstack([
                v_max * np.ones(len(joint_pos_current)*2).reshape(-1, 1)]).reshape(-1)

            q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
            q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            q_nom = (q_max + q_min) / 2.0
            gains = np.ones(7)
            # gains = np.array([40., 30., 50., 25., 35., 25., 10.])
            # gains = np.array([350., 250., 350., 210., 220., 260.,  70.])

            v = cp.Variable(len(joint_pos_current))
            jac_np = jacobian.cpu().numpy()

            constraints = []
            constraints.append(
                G @ v <= h
            )
            constraints.append(
                joint_pos_current.cpu().numpy() + v <= q_max
            )
            constraints.append(
                joint_pos_current.cpu().numpy() + v >= q_min
            )

            error = jac_np @ v - ee_velocity_desired.cpu().numpy().squeeze()
            eps = 1e-2
            # error2 = (np.eye(7) - np.linalg.pinv(jac_np) @ jac_np) @ (v - cp.multiply(gains, (q_nom - joint_pos_current.cpu().numpy())))
            
            # mimicing https://github.com/RobotLocomotion/drake/blob/master/multibody/inverse_kinematics/differential_inverse_kinematics.cc#L118
            J = jac_np
            # _, L, U = scipy.linalg.lu(J)
            # P = scipy.linalg.null_space(U)  # not sure if something faster we can do here # (7x1)
            P = scipy.linalg.null_space(J)  # (7x1)
            # P2 = np.eye(7) - (np.linalg.pinv(J) @ J)  # (7x7)
            v_jc = v - cp.multiply(gains, (q_nom - joint_pos_current.cpu().numpy()))
            projection = P @ (P.T @ v_jc)
            # projection = P2 @ v_jc
            error2 = projection

            # cost = cp.Minimize(cp.norm(error) + eps * cp.norm(error2))
            cost = cp.Minimize(cp.norm(error))

            prob = cp.Problem(cost, constraints)

            prob.solve()
            out = v.value
            return torch.from_numpy(out).float().to(jacobian.device)

        def qp_fwd2(self, joint_pos_current, ee_velocity_desired, jacobian):
            q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
            q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            q_nom = (q_max + q_min) / 2.0
            gains = np.ones(7)
            # gains = np.array([40., 30., 50., 25., 35., 25., 10.])
            # gains = np.array([350., 250., 350., 210., 220., 260.,  70.])

            qdot_max = np.array([2.175] * 7)

            v = cp.Variable(len(joint_pos_current))
            alpha = cp.Variable(1)
            jac_np = jacobian.cpu().numpy()

            constraints = []
            constraints.append(
                jac_np @ v - cp.multiply(alpha, ee_velocity_desired.cpu().numpy()) == 0
            )
            constraints.append(
                alpha <= np.array([1])
            )
            constraints.append(
                alpha >= np.array([0]) 
            )
            constraints.append(
                joint_pos_current.cpu().numpy() + v <= q_max
            )
            constraints.append(
                joint_pos_current.cpu().numpy() + v >= q_min
            )
            constraints.append(
                v <= qdot_max
            )
            constraints.append(
                v >= -1.0 * qdot_max
            )

            error = -1.0 * alpha
            # eps = 1e-4
            # error2 = (np.eye(7) - np.linalg.pinv(jac_np) @ jac_np) @ (v - cp.multiply(gains, (q_nom - joint_pos_current.cpu().numpy())))

            # mimicing https://github.com/RobotLocomotion/drake/blob/master/multibody/inverse_kinematics/differential_inverse_kinematics.cc#L118
            J = jac_np
            v_jc = v - cp.multiply(gains, (q_nom - joint_pos_current.cpu().numpy()))

            # _, L, U = scipy.linalg.lu(J)
            # P = scipy.linalg.null_space(U)  # not sure if something faster we can do here # (7x1)
            # P = scipy.linalg.null_space(J)  # (7x1)
            P2 = np.eye(7) - (np.linalg.pinv(J) @ J)  # (7x7)

            # projection = P @ (P.T @ v_jc)
            projection = P2 @ v_jc

            error2 = projection
            
            # cost = cp.Minimize(100 * error + cp.norm(error2))
            # cost = cp.Minimize(error + eps * cp.norm(error2))
            cost = cp.Minimize(error)
            prob = cp.Problem(cost, constraints)

            prob.solve()
            out = v.value
            return torch.from_numpy(out).float().to(jacobian.device)

        def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            # Get states.
            joint_pos_current = state_dict["joint_positions"]

            # 6x7
            jacobian = state_dict["jacobian_diffik"]

            ee_pos, ee_quat = state_dict["ee_pos"], state_dict["ee_quat"]

            position_error = self.goal_pos - ee_pos
            # quat_error = C.quat_mul(self.goal_ori, C.quat_conjugate(ee_quat))

            ee_mat = C.quat2mat(ee_quat).to(joint_pos_current.device)
            goal_mat = C.quat2mat(self.goal_ori).to(joint_pos_current.device)
            mat_error = torch.matmul(goal_mat, torch.inverse(ee_mat))

            if self.scale_errors:
                # position_error = torch.clamp(position_error, min=-0.01, max=0.01)
                max_pos_value = torch.abs(position_error).max()
                clip_pos_value = 0.01
                if max_pos_value > clip_pos_value:
                    pos_error_scalar = clip_pos_value / max_pos_value
                    position_error = position_error * pos_error_scalar

                # rotvec_error = R.from_quat(quat_error.cpu().numpy()).as_rotvec()
                rotvec_error = R.from_matrix(mat_error.cpu().numpy()).as_rotvec()
                delta_norm = np.linalg.norm(rotvec_error)
                max_rot_radians = 0.04

                # one way to do it, manually
                delta_axis = rotvec_error / delta_norm
                delta_norm_clipped = np.clip(delta_norm, a_min=0.0, a_max=max_rot_radians)
                delta_rotvec_scaled = delta_axis * delta_norm_clipped
                delta_mat = R.from_rotvec(delta_rotvec_scaled).as_matrix() # * -1.0
                mat_error = torch.from_numpy(delta_mat).float().to(joint_pos_current.device)

                # # another way to do it with slerp
                # delta_quat = R.from_rotvec(delta_rotvec_scaled).as_quat() # * -1.0
                # quat_error = torch.from_numpy(delta_quat).float().to(joint_pos_current.device)
                # norm_ratio = max_rot_radians / delta_norm
                # quat_error = C.quat_slerp(torch.Tensor([0., 0., 0., 1.]).float().to(quat_error_pre.device), quat_error_pre, norm_ratio)

            rot_error = R.from_matrix(mat_error.cpu().numpy())

            # position_error = self.ee_pos_error
            # rot_error = self.ee_rot_error

            ee_delta_axis_angle = torch.from_numpy(rot_error.as_rotvec()).float().to(position_error.device)

            dt = 1.0
            ee_pos_vel = position_error * self.pos_scalar / dt
            ee_rot_vel = ee_delta_axis_angle * self.rot_scalar / dt

            ee_velocity_desired = torch.cat((ee_pos_vel, ee_rot_vel), dim=-1)

            if self.use_qp:
                # joint_vel_desired = self.qp_fwd(joint_pos_current, ee_velocity_desired, jacobian)
                joint_vel_desired = self.qp_fwd2(joint_pos_current, ee_velocity_desired, jacobian)
            else:
                joint_vel_desired = torch.linalg.lstsq(jacobian, ee_velocity_desired).solution

            joint_pos_desired = joint_pos_current + joint_vel_desired*dt

            return {"joint_positions": joint_pos_desired}
        
        def set_goal(self, goal_pos, goal_ori):
            self.goal_pos = goal_pos
            self.goal_ori = goal_ori

        def reset(self):
            pass

    return DiffIKController(*args, **kwargs)

