# import json
import torch

import os
import numpy as np
from .augmentations import StrongAug, GeometricAug
import torchvision
from PIL import Image
import torch.nn.functional as F
from .apolloscape.trainId2color import id_list

import cv2 as cv
EST_H = True
if EST_H:
    sift = cv.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    MIN_MATCH_COUNT = 40

With_name = True
Night_flag = False
Stereo = False
Step = 2
ori_size = (2710,3384)
intrinsic_5 = np.array(
        [[2304.54786556982, 0, 1686.23787612802],
         [0, 2305.875668062, 1354.98486439791],
        [0,0,1]])
intrinsic_6 = np.array(
        [[2300.39065314361, 0, 1713.21615190657],
         [0, 2301.31478860597, 1342.91100799715],
        [0,0,1]])
extrinsic_6 = {
    'R': np.array([
        [9.96978057e-01, 3.91718762e-02, -6.70849865e-02],
        [-3.93257593e-02, 9.99225970e-01, -9.74686202e-04],
        [6.69948100e-02, 3.60985263e-03, 9.97746748e-01]]),
    'T': np.array([[-0.6213358], [0.02198739], [-0.01986043]])
}

# id_dict = {0:0, 200:1, 204:2, 213:3, 209:4, 206:5, 207:6, 201:7, 203:8, 211:9, 208:10, 216:11, 217:12,
#             215:13, 218:14, 219:15, 210:16, 232:17, 214:18, 202:19, 220:20, 221:21, 222:22, 231:23,
#             224:24, 225:25, 226:26, 230:27, 228:28, 229:29, 233:30, 205:31, 212:32, 227:33, 223:34, 250:35,
#            249:36, 255:255}
# trainid_dict = {0:0, 1:200, 2:204, 3:213, 4:209, 5:206, 6:207, 7:201, 8:203, 9:211, 10:208, 11:216, 12:217,
#             13:215, 14:218, 15:219, 16:210, 17:232, 18:214, 19:202, 20:220, 21:221, 22:222, 23:231,
#             24:224, 25:225, 26:226, 27:230, 28:228, 29:229, 30:233, 31:205, 32:212, 33:227, 34:223, 35:250,
#            36:249, 255:255}
# id_list = [0, 200, 204, 213, 209, 206, 207, 201, 203, 211, 208, 216, 217,215, 218, 219, 210, 232, 214, 202, 220, 221,
#            222, 231, 224, 225, 226, 230, 228, 229, 233, 205, 212, 227, 223, 250, 249, 255]


ToTensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def decode(img):
    # turn png mask to 1-hot lables
    """
    imput (h, w)
    returns (h, w, n) np.int32 {0, 1}
    """
    h,w = img.shape
    out = np.zeros((len(id_list),h,w))
    for trainid, id in enumerate(id_list):
        out[trainid,:,:] = (img == id)

    return out

def encode(img):
    # turn 1-hot lables to png mask
    """
    input (..., c, h, w)  {0~1} torch
    return (..., h, w)
    """
    out_img = torch.argmax(img, dim=-3)
    # turn train 1d to id
    for trainid, id in enumerate(id_list):
        if trainid == id:
            continue
        out_img = torch.where(out_img==trainid, torch.tensor(id).to(out_img), out_img)
    out_img = out_img.byte()

    return out_img

def get_palettedata():
    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 130, 180, 0, 0, 142, 0, 0, 230,
            119, 11, 32, 220, 20, 60, 255, 128, 0, 0, 0, 60, 0, 60, 100, 0, 0, 160, 255, 0, 0, 128, 64, 128, 244, 35,
            232, 0, 255, 255, 128, 0, 128, 190, 153, 153, 250, 170, 30, 153, 153, 153, 220, 220, 0, 102, 102, 156, 128,
            0, 0, 128, 128, 0, 128, 78, 160, 150, 100, 100, 128, 128, 64, 180, 165, 180, 107, 142, 35, 201, 255, 229,
            178, 132, 190, 51, 255, 51, 250, 128, 114, 0, 191, 255, 255, 165, 0, 238, 232, 170, 127, 255, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 153, 153, 102, 0, 204, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255]


def get_pose_from_file(pose_path):
    pose_lines = np.genfromtxt(pose_path, dtype=(float, float, float, float, float, float, float, float, float,
                                                     float, float, float, float, float, float, float, '|S29'), delimiter=" ")
    pose_dict = {}
    for line in pose_lines:
        # line: E00~E33, file_name
        extrinsic = np.array([[line[0], line[1], line[2], line[3]],[line[4], line[5], line[6], line[7]],
                          [line[8], line[9], line[10], line[11]],[line[12], line[13], line[14], line[15]]])

        pose_dict[str(line[16], 'UTF-8')] = extrinsic
    return pose_dict

def top_crop(img, crop_ratio):
    w, h = img.size
    crop_h = int(crop_ratio*h)
    return torchvision.transforms.functional.crop(img, crop_h, 0, h-crop_h, w) #top,left,height,wide


def EsitmateHomoMatrix(img_ref, img_cur, K, last_norm):
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    kp_cur, des_cur = sift.detectAndCompute(img_cur, None)
    kp_ref, des_ref = sift.detectAndCompute(img_ref, None)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_ref, des_cur, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        cur_pts = np.float32([kp_cur[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv.findHomography(cur_pts, ref_pts, cv.RANSAC, 5.0)
        num, Rs, Ts, Ns = cv.decomposeHomographyMat(H, K)
        # choose the most similar one with last one
        sim_max = 0
        for i in range(num):
            sim = last_norm @ Ns[i]
            if sim > sim_max:
                # max similarity
                N = Ns[i]
                M = np.hstack((Rs[i],Ts[i]))
                M = np.vstack((M,np.array([0,0,0,1])))
                sim_max = sim
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        M = np.eye(4)
        N = last_norm

    return M, N

def calculate_norm(pitch, roll):
    n1 = -np.sin(roll) * np.cos(pitch)
    n2 = -np.cos(roll) * np.cos(pitch)
    n3 = np.sin(pitch)
    return np.array([[n1,n2,n3]])
def homography_trans(image, I_tar, I_src, E, height, N):
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
    # N = calculate_norm(init_pitch, init_roll)

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

    split_file = os.path.join(labels_dir, split+'.txt')
    #records = np.genfromtxt(split_file, dtype=('|S17','|S9', int), delimiter=" ")
    #return [ApolloscapeDataset(dataset_dir, str(s[0][11:], 'UTF-8'), str(s[1], 'UTF-8'), s[2], split, augment, image) for s in records]  # scenes:road02/Record001
    records = np.genfromtxt(split_file, dtype=[('road', 'S17'), ('record', 'S9'), ('camera','i8')], delimiter=" ")
    if records.size <= 1:
        records = [records]

    return [ApolloscapeDataset(dataset_dir, str(s['road'], 'UTF-8')[11:], str(s['record'], 'UTF-8'), s['camera'], split, augment, image) for s in records] #scenes:road02/Record001


class ApolloscapeDataset(torch.utils.data.Dataset):
    """
    Water Hazard Dataset
    # get all information of a scene

    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """
    def __init__(self, dataset_dir, road, record, camera_id, split, augment, image_info):
        self.split = split
        if Night_flag:
            self.img_dir = os.path.join(dataset_dir, 'night','ColorImage_'+road, 'ColorImage', record)
            self.mean = torch.tensor([0.1798, 0.1580, 0.1327])
            self.std = torch.tensor([0.1397, 0.1223, 0.1139])
        else:
            self.img_dir = os.path.join(dataset_dir, 'ColorImage_'+road, 'ColorImage', record)
            self.mean = torch.tensor([0.2830, 0.3077, 0.3341])
            self.std = torch.tensor([0.1698, 0.1841, 0.1898])
        self.mask_dir = os.path.join(dataset_dir, 'Labels_'+road, 'Label', record)
        self.camera_id = camera_id
        # get extrinsic
        pose_txt = os.path.join(dataset_dir, road + '_Pose', record, 'Camera %d'%(camera_id), 'pose.txt')
        self.pose = get_pose_from_file(pose_txt)

        # get file list
        mask_dir = os.path.join(self.mask_dir , 'Camera %d'%(camera_id))
        if not os.path.exists(mask_dir):
            # no mask for test, read form image dir
            mask_dir = os.path.join(self.img_dir, 'Camera %d' % (camera_id))

        self.mask_list = os.listdir(mask_dir)
        self.mask_list.sort()
        # remove no pose items
        for mask_name in self.mask_list:
            image_name = mask_name[:25] + '.jpg'
            if image_name not in self.pose.keys():
                self.mask_list.remove(mask_name)

        if Stereo:
            self.frame_cnt = image_info['sequence_cnt']//2

            # remove no other items
            other_camera = 11-self.camera_id
            other_img_dir = os.path.join(self.img_dir , 'Camera %d'%(other_camera))
            other_imgs = os.listdir(other_img_dir)
            for mask_name in self.mask_list:
                image_name = mask_name[:24] + '%d.jpg'%(other_camera)
                if image_name not in other_imgs:
                    self.mask_list.remove(mask_name)
        else:
            self.frame_cnt = image_info['sequence_cnt']

        resize = [torchvision.transforms.Resize(size=(image_info['h'], image_info['w']))]
        xform = [torchvision.transforms.Resize(size=(image_info['h'], image_info['w']))]+{
            'none': [],
            'strong': [StrongAug()],
            'geometric': [StrongAug(), GeometricAug()],
        }[augment] + [torchvision.transforms.ToTensor()]

        self.img_transform = torchvision.transforms.Compose(xform)
        self.mask_transform = torchvision.transforms.Compose(resize)
        self.top_crop_ratio = image_info['top_crop']
        self.init_pitch = image_info['init_pitch']
        self.init_roll = image_info['init_roll']
        self.camera_height = image_info['camera_height']

        # camera intrinsics, need adjust for size/crop
        self.intrinsics_left = intrinsic_5.copy()
        self.intrinsics_left[1, 2] -= ori_size[0] * image_info['top_crop']
        self.intrinsics_left[0] *= image_info['w'] / ori_size[1]
        self.intrinsics_left[1] *= image_info['h'] / (ori_size[0] * (1-image_info['top_crop']))
        self.intrinsics_right = intrinsic_6.copy()
        self.intrinsics_right[1, 2] -= ori_size[0] * image_info['top_crop']
        self.intrinsics_right[0] *= image_info['w'] / ori_size[1]
        self.intrinsics_right[1] *= image_info['h'] / (ori_size[0] * (1-image_info['top_crop']))
        self.extrinsic_r = np.vstack((np.hstack([extrinsic_6['R'], extrinsic_6['T']]), np.array([0, 0, 0, 1]))) #5(left)->6(right)
        #self.mask_list = self.mask_list[:10] #debug

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        # get images:List[img_path] intrinsics:List[3*3] extrinsics[4*4] scene:"on_road"/"off_road" view:bev3*3
        # pose:4*4 pose_inverse:4*4, cam_ids:List[int], cam_channels: List["left"/'right']
        # no: bev, token, "aux", 'visibility',
        # camera: cam_idx, image, intrinsics, extrinsics

        mask_name = self.mask_list[idx]

        mask = None
        if '.png' in  mask_name:
            # read mask
            mask_path = os.path.join(self.mask_dir , 'Camera %d'%(self.camera_id), mask_name)
            mask = Image.open(mask_path, 'r')
            mask = top_crop(mask, self.top_crop_ratio)
            mask = np.asarray(mask)
            mask = decode(mask) # to 1-hot lable
            # mask = (255 * mask).astype(np.uint8)
            mask = torch.from_numpy(mask)
            mask = self.mask_transform(mask)
        else:
            mask = torch.tensor(int(self.mask_list[idx][7:16]))# used to save timestmp

        # for each frame
        cur_name = self.mask_list[idx][:25]+'.jpg'
        cnt = 0
        frame_num = idx
        images = []
        cam_ids = []
        intrinsics = []
        extrinsics = []
        if EST_H:
            np_frames = []
        while cnt < self.frame_cnt:
            frame_name = self.mask_list[frame_num][:25]+'.jpg'
            frame_path = os.path.join(self.img_dir, 'Camera %d'%(self.camera_id), frame_name)
            frame_cur = Image.open(frame_path, 'r').convert('RGB')
            frame_cur = top_crop(frame_cur, self.top_crop_ratio)
            if EST_H:
                np_frames.append(np.asarray(frame_cur))
            frame_cur = self.img_transform(frame_cur)
            frame_cur = (frame_cur-self.mean[:,None,None])/self.std[:,None,None]
            images.append(frame_cur)
            if self.camera_id == 5:
                intrinsics.append(self.intrinsics_left)
            else:
                intrinsics.append(self.intrinsics_right)
            pose = np.linalg.inv(self.pose[frame_name]) @ self.pose[cur_name] # cur to pre
            extrinsics.append(pose)

            if Stereo:
                # left & right
                cam_ids.append(cnt*2)
                other_camera_id = 11 - self.camera_id
                other_name = self.mask_list[frame_num][:24]+'%d.jpg' % (other_camera_id)
                other_path = os.path.join(self.img_dir, 'Camera %d'%(other_camera_id), other_name)
                frame_other = Image.open(other_path, 'r').convert('RGB')
                frame_other = top_crop(frame_other, self.top_crop_ratio)
                frame_other = self.img_transform(frame_other)
                frame_other = (frame_other - self.mean[:, None, None]) / self.std[:, None, None]
                images.append(frame_other)
                if self.camera_id == 5:
                    intrinsics.append(self.intrinsics_right)
                    extrinsics.append(self.extrinsic_r@pose)
                else:
                    intrinsics.append(self.intrinsics_left)
                    extrinsics.append(np.linalg.inv(self.extrinsic_r) @ pose)
                cam_ids.append(cnt * 2 + 1)
            else:
                # only one
                cam_ids.append(cnt)

            # for next frame
            cnt += 1
            frame_num -= Step
            if frame_num <= 0:
                frame_num = idx + self.frame_cnt*Step - cnt*Step

        if EST_H:  # debug for opencv homo matrix estimation
            extrinsics_es = []
            extrinsics_es.append(np.eye(4))
            norm_list = []
            init_norm = calculate_norm(self.init_pitch, self.init_roll)
            for i in range(1,len(np_frames)):
                M, norm = EsitmateHomoMatrix(np_frames[i], np_frames[0], intrinsics[0], init_norm)
                extrinsics_es.append(M)
                norm_list.append(norm.T)
                init_norm = norm.T
            extrinsics = extrinsics_es
            norm = np.mean(np.array(norm_list), axis=0)
            norm = norm / np.linalg.norm(norm)

        #debug homography transform H = I_tar(R-tn/dis)I_src_inv
        if 0:
            seq_img = None
            for i in range(len(cam_ids)):
                ori = torchvision.transforms.functional.to_pil_image(images[i], mode='RGB')
                ori.save('oriimg_%0d.png' % (i))
                if not EST_H:
                    norm = calculate_norm(self.init_pitch, self.init_roll)
                img = homography_trans(images[i], intrinsics[0], intrinsics[i], extrinsics[i], self.camera_height, norm)
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
                'mask': mask,
            }

        if EST_H:
            out['norm'] = torch.tensor(norm)

        if With_name:
            out['name'] = torch.tensor(int(self.mask_list[idx][7:16]))

        return out

if __name__ == '__main__':
    dataset = get_data(
            '/data/dataset/apolloscape/',
            '/data/dataset/apolloscape/',
            'train',
            0,
            37,
            image={'h': 1024,'w': 3392,'top_crop': 0.62,'sequence_cnt': 2,'norm_start_h': 0.05,'norm_end_h': 0.85,'norm_ignore_w': 0.0}
    )