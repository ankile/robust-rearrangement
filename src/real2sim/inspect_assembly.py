import sys
import os
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
import json

fname1 = sys.argv[1]
fname2 = sys.argv[2]

mesh_moved = trimesh.load(fname1)
mesh_ref = trimesh.load(fname2)

with open("mug_on_rack/assembly.json", "r") as f:
    assembly_data = json.load(f)["data"]

ref_pose_mat = np.asarray(assembly_data["reference"]["pose"]).reshape(4, 4)
moved_pose_mat = np.asarray(assembly_data["moved"]["pose"]).reshape(4, 4)

rel_pose_mat = np.linalg.inv(ref_pose_mat) @ moved_pose_mat

mesh_moved.apply_transform(rel_pose_mat)
scene = trimesh.Scene()
scene.add_geometry(mesh_moved)
scene.add_geometry(mesh_ref)
scene.show()

from IPython import embed

embed()
