import os
import torch
import torch.nn.functional as F
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random
import json


# Dataset with monocular depth and normal
class ETH3DDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 img_res=[640, 960],
                 scan_id=0,
                 num_views=-1):
        scan_to_scene = ['courtyard']
        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        
        with open(self.instance_dir / 'transforms.json') as f:
            trans_json = json.load(f)
        self.n_images = len(trans_json['frames'])

        self.intrinsics_all = []
        self.pose_all = []
        self.rgb_images = []
        self.depth_images = []
        self.normal_images = []


        intrinsic = torch.eye(4)

        if 'fx' in trans_json:
            intrinsic[0, 0] = trans_json['fl_x']
            intrinsic[1, 1] = trans_json['fl_y']
            intrinsic[0, 2] = trans_json['cx']
            intrinsic[1, 2] = trans_json['cy']

        for frame in trans_json['frames']:
            rgb = rend_util.load_rgb(frame['file_path'])
            rgb = rgb.reshape(3, -1).transpose(1, 0)

            normal = rend_util.load_normal(frame['normal_file_path'])
            depth = rend_util.load_depth(frame['depth_file_path'])
            mask = np.ones_like(depth)

            pose = torch.tensor(frame['transform_matrix'])
            pose[:, 1] *= -1
            pose[:, 2] *= -1


            self.rgb_images.append(torch.from_numpy(rgb).float())
            self.normal_images.append(torch.from_numpy(normal).float())
            self.depth_images.append(torch.from_numpy(depth).float())
            self.mask_images.append(torch.from_numpy(mask).float())

            self.intrinsics_all.append(intrinsic)
            self.pose_all.append(pose)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
            ground_truth["full_mask"] = self.mask_images[idx]
         
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.eye(4)
