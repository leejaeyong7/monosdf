import os
import sys
import numpy as np

import torch
from pathlib import Path
import torch.nn.functional as NF
from skimage import measure
from tqdm import tqdm

def extract_sdfs(sdf_q, points, scale, offset, P, fp_16=False, device='cuda'):
    batch_size = 4096 * 4
    with torch.cuda.amp.autocast(enabled=fp_16):
        with torch.no_grad():
            sdfs = []
            for bs in tqdm(range(0, points.shape[0], batch_size), leave=False):
                fp = points[bs:bs+batch_size].float().clone() * scale
                fp = fp + offset[None]

                sdf = sdf_q(fp.float().to(device))
                sdfs.append(sdf.cpu())
            sdfs = torch.cat(sdfs)
    sdfs = sdfs.cpu().float().reshape(P, P, P).numpy()
    return sdfs

def extract_node(sdf_q, points, depth, num_max_d, scale, offset, P, fp_16=False, device='cuda'):
    # compute current resolution
    dx = scale / (P - 1)
    dy = scale / (P - 1)
    dz = scale / (P - 1)

    ox = offset[0].item()
    oy = offset[1].item()
    oz = offset[2].item()

    # compute SDF value
    sdfs = extract_sdfs(sdf_q, points, scale, offset, P, fp_16, device)

    if depth >= num_max_d:
        verts, faces, normals, values = measure.marching_cubes(volume=sdfs, level=0.0, spacing=(dx, dy, dz))
        verts[:, 0] += ox
        verts[:, 1] += oy
        verts[:, 2] += oz
        return verts, faces
    H = int(P // 2)

    all_verts = []
    all_faces = []

    for c in tqdm(range(8), desc=f"Iterating at depth: {depth}", leave=False):
        x = c % 2
        y = (c // 2) % 2
        z = (c // 4) % 2
    

        SX = 0 if x == 0 else H
        EX = H if x == 0 else None
        SY = 0 if y == 0 else H
        EY = H if y == 0 else None
        SZ = 0 if z == 0 else H
        EZ = H if z == 0 else None

        child_sdfs = sdfs[SX:EX, SY:EY, SZ:EZ]
        if (child_sdfs < 0).all() or (child_sdfs > 0).all():
            continue

        cx = ox if x == 0 else ox + scale / 2.0
        cy = oy if y == 0 else oy + scale / 2.0
        cz = oz if z == 0 else oz + scale / 2.0

        child_offset = torch.tensor([cx, cy, cz]).to(points)

        verts, faces = extract_node(sdf_q, points, depth + 1, num_max_d, scale / 2.0, child_offset, P, fp_16, device)
        all_verts.append(verts)
        all_faces.append(faces)

    verts = []
    faces = []
    vert_i = 0
    for all_vert, all_face in zip(all_verts, all_faces):
        verts.append(all_vert)
        faces.append(all_face + vert_i)
        vert_i += len(all_vert)
    verts = np.concatenate(verts)
    faces = np.concatenate(faces)

    return verts, faces


def extract_mesh_tree(sdf_q, grid_boundary, device='cuda', fp_16=True, num_d=2):
    aabb = torch.tensor([
        grid_boundary[0],
        grid_boundary[0],
        grid_boundary[0],
        grid_boundary[1],
        grid_boundary[1],
        grid_boundary[1]
    ])

    min_xyz = aabb[:3]
    max_xyz = aabb[3:]
    scale = (max_xyz - min_xyz).max().item()
    resolution = 256

    # compute root node
    X, Y, Z = np.mgrid[:resolution, :resolution, :resolution]
    # points range in aabb
    points = torch.from_numpy(np.stack((X, Y, Z), -1)) / (resolution - 1)
    points = points.view(-1, 3)
    level = 0

    verts, faces = extract_node(sdf_q, points, level, num_d, scale, min_xyz.to(points), resolution, fp_16, device)
    return verts, faces
    