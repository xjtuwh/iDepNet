import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import cv2
import os
import pickle as pickle
from kitti.kitti_object import *
from kitti.kitti_helper import Kitti_Config
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def rotate_pc_along_y(pc, rot_angle):
    """
    params pc: (N, 3+C), (N, 3) is in the rectified camera coordinate
    params rot_angle: rad scalar
    Output pc: updated pc with XYZ rotated
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc

def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio
    xmin, ymin, xmax, ymax = box2d
    h = ymax - ymin
    w = xmax - xmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cx2 = cx + w * r * (np.random.random() * 2 - 1)
    cy2 = cy + h * r * (np.random.random() * 2 - 1)
    h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    return np.array([cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return np.load(path).astype(np.float32)


def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def kitti2015_disparity_loader(filepath, calib):
    disp = np.array(Image.open(filepath))/256.
    depth = np.zeros_like(disp)
    mask = disp > 0
    depth[mask] = calib / disp[mask]
    return depth


def dynamic_baseline(calib_info):
    P3 =np.reshape(calib_info['P3'], [3,4])
    P =np.reshape(calib_info['P2'], [3,4])
    baseline = P3[0,3]/(-P3[0,0]) - P[0,3]/(-P[0,0])
    return baseline

class myImageFloder(data.Dataset):
    def __init__(self, data, training, split, kitti2015=False, dynamic_bs=False, loader=default_loader, dploader=disparity_loader):
        overwritten_data_path = os.path.join('/media/zd/2T/jcf/psmnet2_patch/',
                                'kitti/frustum_caronly_psmnet_%s.pickle' % (split))
        with open(overwritten_data_path, 'rb') as fp:
            self.id_list = pickle.load(fp, encoding='iso-8859-1')
            self.box2d_list = pickle.load(fp, encoding='iso-8859-1')
            # self.left_img_list = pickle.load(fp, encoding='iso-8859-1')
            # self.right_img_list = pickle.load(fp, encoding='iso-8859-1')
            # self.seg_mask_list = pickle.load(fp, encoding='iso-8859-1')
            # self.depth_list = pickle.load(fp, encoding='iso-8859-1')
            self.depth_min_list = pickle.load(fp, encoding='iso-8859-1')
            self.l_box2d_list = pickle.load(fp, encoding='iso-8859-1')
            self.mask_list = pickle.load(fp, encoding='iso-8859-1')
            self.box3d_gt_list = pickle.load(fp, encoding='iso-8859-1')
            self.frustum_angle_list = pickle.load(fp, encoding='iso-8859-1')

        self.dataset = kitti_object('/media/zd/2T/jcf/psmnet2_min/dataset/KITTI/object', 'training')
        left, right, left_depth, left_calib, seg_mask, label = data
        self.left = left
        self.dynamic_bs = dynamic_bs
        self.right = right
        self.depth = left_depth
        self.calib = left_calib
        self.label = label
        self.seg_mask = seg_mask
        self.loader = loader
        self.kitti2015 = kitti2015
        self.dploader = dploader
        self.training = training
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ])
        self.dataset_helper = Kitti_Config()
        self.rotate_to_center = True
        # self.box3d_center_list = self.box3d_gt_list
        # self.box3d_size_list =
        # self.heading_list =


    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it is shifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + self.frustum_angle_list[index]

    def patchnet_pre(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)
        box3d_gt = self.box3d_gt_list[index]
        cls_type = 'Car'
        assert(cls_type in ['Car', 'Pedestrian', 'Cyclist'])
        one_hot_vec = np.zeros((3), dtype=np.float32)
        one_hot_vec[self.dataset_helper.type2onehot[cls_type]] = 1

        # if self.from_rgb_detection:
        #     return patch, rot_angle, self.prob_list[index], self.id_list[index], \
        #            self.type_list[index], self.box2d_list[index], one_hot_vec

        # ------------------------------ LABELS ----------------------------
        # size labels
        size = box3d_gt[3:6]
        size_class, size_residual = self.dataset_helper.size2class(size, 'Car')

        # center labels
        center = box3d_gt[0:3]
        center = center-np.array([0,size[2]/2,0]).astype(np.float32)
        if self.rotate_to_center:
            center = rotate_pc_along_y(np.expand_dims(center,0), self.get_center_view_rot_angle(index)).squeeze()

        # heading labels
        heading_angle = box3d_gt[6]
        if self.rotate_to_center:
            heading_angle = heading_angle - rot_angle

        angle_class, angle_residual = self.dataset_helper.angle2class(heading_angle)

        return center, angle_class, angle_residual, size_class, size_residual, rot_angle, one_hot_vec

    def __getitem__(self, index):
        data_idx = self.id_list[index]
        unit_box2d = self.box2d_list[index]
        left_box2d = self.l_box2d_list[index]
        depth_gt = self.depth_min_list[index]
        mask_number = self.mask_list[index]
        depth_disturb = random.uniform(depth_gt - 3, depth_gt + 3)
        left_img = self.dataset.get_image(data_idx)
        W, H = left_img.size
        right_img = self.dataset.get_right_image(data_idx)
        depth = self.dataset.get_depth(data_idx)
        depth = np.array(depth).astype(np.float32) / 256
        seg_mask = self.dataset.get_mask(data_idx)
        seg_mask[seg_mask > 0] = 1
        # mask_image = self.dataset.get_mask(data_idx)
        # seg_mask = (mask_image == mask_number)

        calib_info = self.dataset.get_calibration(data_idx)
        xmin, ymin, xmax, ymax = random_shift_box2d(unit_box2d)
        # xmin, ymin, xmax, ymax = unit_box2d
        xmin, ymin = max(xmin, 0), max(ymin, 0)   # check range
        xmax, ymax = min(xmax, W), min(ymax, H)   # check range
        xmin, ymin, xmax, ymax = unit_box2d
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        unit_box2d = np.array([xmin, ymin, xmax, ymax])

        xmin1, ymin1, xmax1, ymax1 = random_shift_box2d(left_box2d)
        xmin1=max(xmin1,xmin)
        ymin1, xmax1, ymax1 =ymin, xmax, ymax
        xmin1 = int(xmin1)
        ymin1 = int(ymin1)
        xmax1 = int(xmax1)
        ymax1 = int(ymax1)
        left_box2d = np.array([xmin1, ymin1, xmax1, ymax1])

        object_mask = np.zeros((int(ymax)-int(ymin),int(xmax)-int(xmin)))
        temp = seg_mask[int(ymin):int(ymax), int(xmin):int(xmax)]
        object_mask[:, int(xmin1) - int(xmin):] = temp[:, int(xmin1) - int(xmin):]
        object_depth = depth[int(ymin):int(ymax), int(xmin):int(xmax)]
        object_mask = object_mask*(object_depth>0)
        object_left_img = left_img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
        object_right_img = right_img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
        # depth_min = calib_info.P[0, 0] * 0.54 / (unit_box2d[2] - unit_box2d[0])
        depth_min = max(2.0,(depth_disturb-10) / 224 * (int(xmax) - int(xmin)))
        depth_max= min(40.0,(depth_disturb+10) / 224 * (int(xmax) - int(xmin)))
        object_left_img = self.transform(object_left_img)
        object_right_img = self.transform(object_right_img)
        object_mask = torch.from_numpy(object_mask).float()  # H * W * C
        object_mask = object_mask.unsqueeze(0)
        object_mask = object_mask.unsqueeze(0)
        object_mask = F.interpolate(object_mask, (224,224), None,mode='bilinear', align_corners=False).squeeze()
        object_depth = torch.from_numpy(object_depth).float()
        # H * W * C
        object_depth = object_depth.unsqueeze(0)
        object_depth = object_depth.unsqueeze(0)
        object_depth = F.interpolate(object_depth, (224, 224), mode='bilinear', align_corners=True).squeeze()
        object_depth = object_depth / 224 * (int(xmax) - int(xmin))
        object_mask1 = object_depth>depth_min
        object_mask2 = object_depth<depth_max
        object_mask = object_mask*object_mask1*object_mask2
        if self.dynamic_bs:
            calib = calib_info.P[0, 0] * 0.54
        else:
            # calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * 0.54
            calib = calib_info.P[0, 0] * 0.54
        #-------------------------------patchnet-----------------------------------------------------
        box_label = {}
        box_label['center'], box_label['angle_class'], box_label['angle_residual'], \
        box_label['size_class'], box_label['size_residual'], \
        box_label['rot_angle'], box_label['one_hot_vec'] = self.patchnet_pre(index)
        # box3d_gt = self.box3d_gt_list[index]
        # rot_angle = self.get_center_view_rot_angle(index)


        # left_img = self.loader(left)
        # right_img = self.loader(right)
        # if self.kitti2015:
        #     dataL = kitti2015_disparity_loader(depth, calib)
        # else:
        #     dataL = self.dploader(depth)
        # instance_image = cv2.imread(seg_image_path, cv2.IMREAD_GRAYSCALE)
        # instance_image[instance_image > 0] = 1
        # W, H = left_img.size
        # top_pad = 384 - H
        # right_pad = 1280 - W
        # left_img = self.transform(left_img)
        # right_img = self.transform(right_img)
        # dataL = torch.from_numpy(dataL).float()
        # instance_image = torch.from_numpy(instance_image).float()
        # left_img = F.pad(left_img, (0, right_pad, 0, top_pad), "constant", 0)    #left,right top, bottom
        # right_img = F.pad(right_img, (0, right_pad,0,top_pad), "constant", 0)
        # dataL = F.pad(dataL, (0, right_pad, 0, top_pad), "constant", -1)
        # instance_image = F.pad(instance_image, (0, right_pad, 0, top_pad), "constant", -1)
        # left_img = left_img.unsqueeze(0)
        # right_img = right_img.unsqueeze(0)
        # dataL = dataL.unsqueeze(0)
        # dataL = dataL.unsqueeze(0)
        # instance_image = instance_image.unsqueeze(0)
        # instance_image = instance_image.unsqueeze(0)
        # left_img=F.interpolate(left_img, size=(384//2, 1280//2))
        # right_img = F.interpolate(right_img, size=(384 // 2, 1280 // 2))
        # dataL = F.interpolate(dataL, size=(384 // 2, 1280 // 2))
        # instance_image = F.interpolate(instance_image, size=(384 // 2, 1280 // 2))
        # left_img = left_img.squeeze(0)
        # right_img = right_img.squeeze(0)
        # dataL = dataL.squeeze(0)
        # dataL = dataL.squeeze(0)
        # instance_image = instance_image.squeeze(0)
        # instance_image = instance_image.squeeze(0)
        # if self.training:
        #     w, h = left_img.size
        #     th, tw = 256, 512
        #
        #     x1 = random.randint(0, w - tw)
        #     y1 = random.randint(0, h - th)
        #
        #     left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
        #     right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
        #
        #     dataL = dataL[y1:y1 + th, x1:x1 + tw]
        #
        #     left_img = self.transform(left_img)
        #     right_img = self.transform(right_img)
        #
        # else:
        #     w, h = left_img.size
        #
        #     # left_img = left_img.crop((w - 1232, h - 368, w, h))
        #     # right_img = right_img.crop((w - 1232, h - 368, w, h))
        #     left_img = left_img.crop((w - 1200, h - 352, w, h))
        #     right_img = right_img.crop((w - 1200, h - 352, w, h))
        #     w1, h1 = left_img.size
        #
        #     # dataL1 = dataL[h - 368:h, w - 1232:w]
        #     dataL = dataL[h - 352:h, w - 1200:w]
        #
        #     left_img = self.transform(left_img)
        #     right_img = self.transform(right_img)
        #
        # dataL = torch.from_numpy(dataL).float()
        return object_left_img.float(), object_right_img.float(), object_depth.float(), \
               object_mask.int(), calib.item(), np.array([depth_min,depth_max]),unit_box2d,\
               left_box2d,calib_info.P,box_label

    def __len__(self):
        return len(self.id_list)
