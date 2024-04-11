# import json
import torch

from pathlib import Path
from .common import get_split
from .transforms import Sample, LoadDataTransform
import re
import os
import numpy as np
# import cv2
from .water_hazard.zedutils import getTransformFromConfig, getKRTInfo
from .water_hazard.transformations import compose_matrix,euler_from_quaternion
from .augmentations import StrongAug, GeometricAug
import torchvision
import cv2
from PIL import Image
import torch.nn.functional as F

Debug = False
With_name = True
Target = 'off_road' #'both_road''off_road' #
Stereo = False
# map1x, map1y, map2x, map2y, K, Q1 = getTransformFromConfig(Path(__file__).parent/'water_hazard'/'SN1994.conf', Type='CAM_HD')
K_l, K_r, R_lr, T_lr= getKRTInfo(Path(__file__).parent/'water_hazard'/'SN1994.conf', Type='CAM_HD') # left->right
ori_size = (720, 1280)

ToTensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def get_pose_from_file(pose_path, timestamp_path):
    timestamps = np.genfromtxt(timestamp_path, dtype=float)
    timestamp_dict = {int(stamp*1e6): id+1 for id, stamp in enumerate(timestamps)}
    pose_lines = np.genfromtxt(pose_path, dtype=(float, float, float, float, float, float, float, float),
                               delimiter=" ")
    pose_dict = {}
    for line in pose_lines:
        # line: timestamp, tx,ty,tz,rx,ry,rz
        # timestamp to idx
        key = timestamp_dict[int(np.around(line[0] / 1e3))]
        # R and T to matrix M
        trans = np.array([line[1], line[2], line[3]])
        angles = euler_from_quaternion(np.array([line[7], line[4], line[5], line[6]]))
        extrinsic = compose_matrix(angles=angles, translate=trans)

        pose_dict[key] = extrinsic

    return pose_dict

def top_crop(img, crop_ratio):
    w, h = img.size
    crop_h = int(crop_ratio*h)
    return torchvision.transforms.functional.crop(img, crop_h, 0, h-crop_h, w) #top,left,height,wide

def left_right_crop(img, left):
    w, h = img.size
    crop_w = w//2
    if left:
        return torchvision.transforms.functional.crop(img, 0, 0, h, crop_w) #top,left,height,wide
    else:
        return torchvision.transforms.functional.crop(img, 0, crop_w, h, w-crop_w)

def calculate_norm(pitch, roll):
    n1 = -np.sin(roll) * np.cos(pitch)
    n2 = -np.cos(roll) * np.cos(pitch)
    n3 = np.sin(pitch)
    return np.array([[n1,n2,n3]])
def homography_trans(image, I_tar, I_src, E, height, init_pitch, init_roll):
    # inputs:
    #   image: src image
    #   I_tar,I_src: 3*3 K matrix
    #   E: rotation(src->tar) and translate(tar->src in tar)
    #   tar_R: rotation of target camera
    # return:
    #   out: tar image

    c, h, w = image.shape

    # get back warp matrix
    i = torch.arange(0, h)
    j = torch.arange(0, w)
    ii, jj = torch.meshgrid(i, j)  # i:h,j:w
    ones = torch.ones_like(ii)
    uv1 = torch.stack([jj, ii, ones], dim=-1).float()  # shape = [h, w, 3]

    # calculate n_vec
    #N = np.array([[0, -1, 0]])  # y axis
    N = calculate_norm(init_pitch, init_roll)

    # H = K(R-tn/d)inv(K)
    R = E[:3, :3]
    T = E[:3, -1:]
    H = I_src @ (R - (T @ N / height)) @ np.linalg.inv(I_tar)
    H = torch.from_numpy(H).float()

    # project to camera
    uv1 = torch.einsum('ij, hwj -> hwi', H, uv1)  # shape = [h,w,3]
    # only need view in front of camera ,Epsilon = 1e-6
    uv_last = torch.maximum(uv1[:, :, 2:], torch.ones_like(uv1[:, :, 2:]) * 1e-6)
    uv = uv1[:, :, :2] / uv_last  # shape = [h,w,2]

    # lefttop to center
    uv_center = uv - torch.tensor([w // 2, h // 2])  # shape = [h,w,2]
    # u:south, v: up from center to -1,-1 top left, 1,1 buttom right
    scale = torch.tensor([w // 2, h // 2])
    uv_center /= scale

    out = F.grid_sample(image.unsqueeze(0), uv_center.unsqueeze(0), mode='bilinear',
                        padding_mode='zeros')

    if c == 1:
        mode = 'L'
    else:
        mode = 'RGB'
    out = torchvision.transforms.functional.to_pil_image(out.squeeze(0), mode=mode)
    return out

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    num_classes,
    augment='none',
    image=None,                         # image config
    dataset='unused',                   # ignore
    **dataset_kwargs
):
    # Override augment if not training
    augment = 'none' if split != 'train' else augment

    # get on/off scene
    if Target == 'off_road':
        split_scenes = ['off_road']
    elif Target == 'on_road':
        split_scenes = ['on_road']
    else:
        split_scenes = ['off_road','on_road']

    if split == 'val':
        split = 'test'

    return [WaterHazardDataset(s, labels_dir, split, augment, image) for s in split_scenes]


class WaterHazardDataset(torch.utils.data.Dataset):
    """
    Water Hazard Dataset
    # get all information of a scene

    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """
    def __init__(self, scene_name, labels_dir, split, augment, image_info):
        self.scene = scene_name
        split_dir = Path(__file__).parent / 'splits' / 'water_hazard'
        split_path = split_dir / f'{scene_name}_{split}.txt'
        files = np.genfromtxt(split_path, dtype='str')
        self.files = files[:,1] # get mask address
        self.labels_dir = labels_dir
        if scene_name == 'off_road':
            self.left_mean = torch.tensor([0.2922, 0.2025, 0.1327])
            self.left_std = torch.tensor([0.0879, 0.0651, 0.0591])
            self.right_mean = torch.tensor([0.2251, 0.1475, 0.0839])
            self.right_std = torch.tensor([0.0697, 0.0517, 0.0378])
        else:
            self.left_mean = torch.tensor([0.1734, 0.1565, 0.1149])
            self.left_std = torch.tensor([0.0624, 0.0590, 0.0611])
            self.right_mean = torch.tensor([0.1274, 0.1111, 0.0744])
            self.right_std = torch.tensor([0.0429, 0.0426, 0.0392])
        if Stereo:
            self.frame_cnt = image_info.sequence_cnt//2
        else:
            self.frame_cnt = image_info.sequence_cnt

        resize = [torchvision.transforms.Resize(size=(image_info.h, image_info.w)),
            torchvision.transforms.ToTensor()]
        xform = [torchvision.transforms.Resize(size=(image_info.h, image_info.w))]+{
            'none': [],
            'strong': [StrongAug()],
            'geometric': [StrongAug(), GeometricAug()],
        }[augment] + [torchvision.transforms.ToTensor()]

        self.img_transform = torchvision.transforms.Compose(xform)
        self.mask_transform = torchvision.transforms.Compose(resize)
        self.top_crop_ratio = image_info.top_crop

        # camera intrinsics, need adjust for size/crop
        self.intrinsics_left = K_l.copy()
        self.intrinsics_left[1, 2] -= ori_size[0] * image_info.top_crop
        self.intrinsics_left[0] *= image_info.w / ori_size[1]
        self.intrinsics_left[1] *= image_info.h / (ori_size[0]*(1-image_info.top_crop))
        self.intrinsics_right = K_r.copy()
        self.intrinsics_right[1, 2] -= ori_size[0] * image_info.top_crop
        self.intrinsics_right[0] *= image_info.w / ori_size[1]
        self.intrinsics_right[1] *= image_info.h / (ori_size[0]*(1-image_info.top_crop))
        self.extrinsic_r = np.vstack((np.hstack([R_lr, (T_lr/1000)[:, None]]), np.array([0, 0, 0, 1]))) #left->right

        pose_file_path = os.path.join(self.labels_dir, 'camera_pose', self.scene + '_camerapose.txt')
        time_stamp_path = os.path.join(self.labels_dir, 'video_'+self.scene, 'TimeStamp.txt')
        self.pose = get_pose_from_file(pose_file_path, time_stamp_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # for i in range(len(self.files)):
        #     if self.files[i][-13:-4] == '000001908':
        #         idx = i
        #         break
        file_name = self.files[idx]
        file_num = int(re.findall('\d+', file_name)[0])

        # read mask
        mask_path = os.path.join(self.labels_dir, 'masks', self.scene, 'left_mask_%09d.png' % (file_num))
        mask = Image.open(mask_path, 'r').convert('L')
        mask = top_crop(mask, self.top_crop_ratio)
        mask = self.mask_transform(mask) # mask value not right
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        # for each frame
        cnt = 0
        frame_num = file_num
        images = []
        cam_ids = []
        intrinsics = []
        extrinsics = []
        while cnt < self.frame_cnt:
            img_path = os.path.join(self.labels_dir, 'video_'+self.scene, 'img_%09d.ppm' % (frame_num))
            pair = Image.open(img_path, 'r').convert('RGB')
            pair = top_crop(pair, self.top_crop_ratio)
            frame_left = left_right_crop(pair, True)
            frame_left = self.img_transform(frame_left)
            if not Debug:
                frame_left = (frame_left-self.left_mean[:,None,None])/self.left_std[:,None,None]
            images.append(frame_left)
            intrinsics.append(self.intrinsics_left)
            pose = np.linalg.inv(self.pose[frame_num]) @ self.pose[file_num] # cur->pre
            extrinsics.append(pose)

            if Stereo:
                # left & right
                cam_ids.append(cnt*2)
                frame_right = left_right_crop(pair, False)
                frame_right = self.img_transform(frame_right)
                if not Debug:
                    frame_right = (frame_right - self.right_mean[:, None, None]) / self.right_std[:, None, None]
                images.append(frame_right)
                intrinsics.append(self.intrinsics_right)
                extrinsics.append(self.extrinsic_r@pose)
                cam_ids.append(cnt * 2 + 1)
            else:
                # only left
                cam_ids.append(cnt)

            # for next frame
            cnt += 1
            frame_num -= 1
            if frame_num <= 0:
                frame_num = file_num + self.frame_cnt - cnt

        #debug homography transform H = I_tar(R-tn/dis)I_src_inv
        if 0:
            seq_img = None
            for i in range(len(cam_ids)):
                img = homography_trans(images[i], intrinsics[0], intrinsics[i], extrinsics[i], self.camera_height, self.init_pitch, self.init_roll)
                if i % 2 == 1 and Stereo:
                    img.save('img_r_%0d.png' % (i))
                    right_img = img.convert("RGBA")
                    lr_img = Image.blend(left_img, right_img, alpha=.5)
                    lr_img.save('img_rl_%0d.png' % (i))
                else:
                    img.save('img_l_%0d.png' % (i))
                    left_img = img.convert("RGBA")
                    if seq_img is not None:
                        seq_img = Image.blend(seq_img, left_img, alpha=.5)
                        seq_img.save('img_l_seq_%0d.png' % (i))
                    else:
                        seq_img = left_img

        # turn list to np at first
        cam_ids = np.array(cam_ids)
        intrinsics = np.array(intrinsics)
        extrinsics = np.array(extrinsics)

        out = {
            'cam_idx': torch.tensor(cam_ids),
            'image': torch.stack(images, 0),
            'intrinsics': torch.tensor(intrinsics),
            'extrinsics': torch.tensor(extrinsics),
            'mask': mask
        }

        if With_name:
            out['name'] = torch.tensor(file_num)

        return out

if __name__ == '__main__':
    dataset = get_data(
            '/data/dataset/water_hazard/',
            '/data/dataset/water_hazard/',
            'train',
            0,
            1,
            image={'h': 160,'w': 640,'top_crop': 0.5,'sequence_cnt': 2,'norm_start_h': 0.08,'norm_end_h': 0.90,
                   'norm_ignore_w': 0.165, 'init_pitch': 0.0, 'init_roll': 0.00, 'camera_height': 1.77})
