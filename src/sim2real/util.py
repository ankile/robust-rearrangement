import os.path as osp
import asyncio
import six
import numpy as np
import torch
import yourdfpy
import trimesh
from scipy.spatial.transform import Rotation as R

from omni.isaac.shapenet.shape import convert


class URDFWithCanonPCDS:
    def __init__(self, urdf):
        self.urdf = urdf
        self.canonical_link_pcds = None

    def __getattr__(self, name):
        return getattr(self.urdf, name)

    def build_canon_robot_pcds(self, n_pts_per_link=1024):
        canonical_link_mesh_fnames = []
        canonical_link_mesh_meshes = []
        # canonical_link_pcds = []
        canonical_link_pcds = {}

        # for scene_link_name, scene_link_geom in urdf.scene.geometry.items():
        # link_fname = scene_link_geom.metadata['file_path']
        for link_name, link in self.link_map.items():
            link_fname = self._filename_handler(
                fname=link.visuals[0].geometry.mesh.filename
            )
            # link_fname = urdf._filename_handler(fname=link.collisions[0].geometry.mesh.filename)
            link_mesh = trimesh.load(
                link_fname, ignore_broken=True, force="mesh", skip_materials=True
            )
            link_pcd = link_mesh.sample(n_pts_per_link)

            canonical_link_mesh_fnames.append(link_fname)
            canonical_link_mesh_meshes.append(link_mesh)
            # canonical_link_pcds.append(link_pcd)
            canonical_link_pcds[link_name] = link_pcd

        self.canonical_link_pcds = canonical_link_pcds


def create_panda_urdf():

    urdf = URDFWithCanonPCDS(
        yourdfpy.URDF.load(
            "assets/franka/panda_with_gripper.urdf",
            build_collision_scene_graph=True,
            load_collision_meshes=True,
        )
    )
    urdf._create_scene(
        use_collision_geometry=False, force_mesh=True
    )  # makes trimesh.Scene object
    urdf.build_canon_robot_pcds()

    return urdf


def transform_pcd(pcd: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if pcd.shape[1] != 4:
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new


def compute_world_poses(
    urdf: URDFWithCanonPCDS, joint_values: np.ndarray, type: str = "numpy"
):
    jnt = joint_values
    # jnt = joint_values[:-1]
    # jnt[-1] = jnt[-1]*0.02 + 0.02

    # urdf.update_cfg(jnt) # This has to match the n-dof of actuated joints
    configuration = jnt
    joint_cfg = []
    if isinstance(configuration, dict):
        for joint in configuration:
            if isinstance(joint, six.string_types):
                joint_cfg.append((urdf._joint_map[joint], configuration[joint]))
            elif isinstance(joint, urdf.Joint):
                # TODO: Joint is not hashable; so this branch will not succeed
                joint_cfg.append((joint, configuration[joint]))
    elif isinstance(configuration, (list, tuple, np.ndarray)):
        if len(configuration) == len(urdf.robot.joints):
            for joint, value in zip(urdf.robot.joints, configuration):
                joint_cfg.append((joint, value))
        elif len(configuration) == urdf.num_actuated_joints:
            for joint, value in zip(urdf._actuated_joints, configuration):
                joint_cfg.append((joint, value))
        else:
            raise ValueError(
                f"Dimensionality of configuration ({len(configuration)}) doesn't match number of all ({len(urdf.robot.joints)}) or actuated joints ({urdf.num_actuated_joints})."
            )
    else:
        raise TypeError("Invalid type for configuration")

    tf_pcd_list = []
    world_pose_list = []
    for j, q in joint_cfg + [
        (j, 0.0) for j in urdf.robot.joints if j.mimic is not None
    ]:
        matrix, joint_q = urdf._forward_kinematics_joint(j, q=q)

        # update internal configuration vector - only consider actuated joints
        if j.name in urdf.actuated_joint_names:
            urdf._cfg[
                urdf.actuated_dof_indices[urdf.actuated_joint_names.index(j.name)]
            ] = joint_q

        # print(f'Matrix: {matrix}, q: {q}')

        # update internal configuration vector - only consider actuated joints
        if j.name in urdf.actuated_joint_names:
            urdf._cfg[
                urdf.actuated_dof_indices[urdf.actuated_joint_names.index(j.name)]
            ] = joint_q

        if urdf._scene is not None:
            urdf._scene.graph.update(
                frame_from=j.parent, frame_to=j.child, matrix=matrix
            )
        if urdf._scene_collision is not None:
            urdf._scene_collision.graph.update(
                frame_from=j.parent, frame_to=j.child, matrix=matrix
            )

        world_pose = urdf.scene.graph.get(
            frame_to=j.child, frame_from=urdf.scene.graph.base_frame
        )[0]
        # if j.child == "panda_rightfinger":
        #     # flip the last finger around
        #     zrot = R.from_euler('xyz', [0, 0, np.pi]).as_matrix()
        #     ztf = np.eye(4); ztf[:-1, :-1] = zrot
        #     world_pose = np.matmul(world_pose, ztf)
        # print(f'World pose: {world_pose}, joint: {j.child}')

        tf_pcd = transform_pcd(urdf.canonical_link_pcds[j.child], world_pose)
        tf_pcd_list.append(tf_pcd)

        if type == "torch":
            world_pose_list.append(torch.from_numpy(world_pose.copy()).float())
        else:
            world_pose_list.append(world_pose)

    return world_pose_list, tf_pcd_list


def convert_obj2usd(
    obj_file: str, overwrite_exists: bool = False, out_type: str = ".usd"
) -> None:

    # Convert obj to usd
    usd_file = obj_file.replace(".obj", out_type)
    print(f"Converting obj file: {obj_file} to USD file: {usd_file}")

    if osp.exists(usd_file) and not overwrite_exists:
        print(f"USD file {usd_file} already exists")
    else:
        print(f"Creating {usd_file}")
        # This is needed to make the convert blocking
        asyncio.get_event_loop().run_until_complete(convert(obj_file, usd_file))

    return usd_file


if __name__ == "__main__":
    urdf = create_panda_urdf()

    # need joints to be number of actuated joints (8 for the franka - last one is mimiced)
    joint_pos = np.array([-0.1363, -0.0406, -0.0460, -2.1322, 0.0191, 2.0759, 0.5, 0.0])
    urdf.update_cfg(joint_pos)

    print(f"Here with urdf")
    from IPython import embed

    embed()

    # all joints (9?)
    joint_pos = np.array(
        [-0.1363, -0.0406, -0.0460, -2.1322, 0.0191, 2.0759, 0.5, 0.0, 0.0]
    )
    world_poses = compute_world_poses(urdf, joint_pos)
