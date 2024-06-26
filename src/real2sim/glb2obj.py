import os, os.path as osp
import argparse

import trimesh

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out-dir", type=str, required=True)
parser.add_argument("-f", "--file", type=str, required=True)
parser.add_argument("-d", "--dry-run", action="store_true")

args = parser.parse_args()

assert args.file.endswith(".glb"), f"Currently only accepts files ending in .glb, filename: {args.file}"
mesh = trimesh.load(args.file)
os.makedirs(args.out_dir, exist_ok=True)
out_fname = osp.join(args.out_dir, args.file.split("/")[-1].replace(".glb", ".obj"))
print(f"Saving to {out_fname}")
if not args.dry_run:
    mesh.export(out_fname)
