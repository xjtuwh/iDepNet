import os
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np
import random
from kitti.kitti_object import *
import pickle as pickle
import torch
from kitti.kitti_helper import Kitti_Config
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


def dataloader(filepath, split):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    calib_fold = 'calib/'
    with open(split, 'r') as f:
        image = [x.strip() for x in f.readlines() if len(x.strip())>0]

    left_test = [filepath + left_fold + img + '.png' for img in image]
    right_test = [filepath + right_fold + img + '.png' for img in image]
    calib_test = [filepath + calib_fold + img + '.txt' for img in image]

    return left_test, right_test, calib_test

def dynamic_baseline(calib_info):
    P3 =np.reshape(calib_info['P3'], [3,4])
    P =np.reshape(calib_info['P2'], [3,4])
    baseline = P3[0,3]/(-P3[0,0]) - P[0,3]/(-P[0,0])
    return baseline

class SubmiteDataset(object):
    def __init__(self, filepath, name, split, dynamic_bs=False, kitti2015=False):
        overwritten_data_path = os.path.join('/media/zd/2T/jcf/psmnet2_min/',
                                             'kitti/frustum_caronly_psmnet_%s.pickle' % (name))
        with open(overwritten_data_path, 'rb') as fp:
            self.id_list = pickle.load(fp, encoding='iso-8859-1')
            self.box2d_list = pickle.load(fp, encoding='iso-8859-1')
            self.box3d_gt_list = pickle.load(fp, encoding='iso-8859-1')
            # self.left_img_list = pickle.load(fp, encoding='iso-8859-1')
            # self.right_img_list = pickle.load(fp, encoding='iso-8859-1')
            # self.seg_mask_list = pickle.load(fp, encoding='iso-8859-1')
            # self.depth_list = pickle.load(fp, encoding='iso-8859-1')
            self.depth_min_list = pickle.load(fp, encoding='iso-8859-1')
            self.left_box2d_list = pickle.load(fp, encoding='iso-8859-1')

        self.dataset = kitti_object('/media/zd/2T/jcf/psmnet2_min/dataset/KITTI/object', 'training')

        self.dynamic_bs = dynamic_bs
        left_fold = 'image_2/'
        right_fold = 'image_3/'
        calib_fold = 'calib/'
        with open(split, 'r') as f:
            image = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        image = sorted(image)

        if kitti2015:
            self.left_test = [filepath + '/' + left_fold + img + '_10.png' for img in image]
            self.right_test = [filepath + '/' + right_fold + img + '_10.png' for img in image]
            self.calib_test = [filepath + '/' + calib_fold + img + '.txt' for img in image]
        else:
            self.left_test = [filepath + '/' + left_fold + img + '.png' for img in image]
            self.right_test = [filepath + '/' + right_fold + img + '.png' for img in image]
            self.calib_test = [filepath + '/' + calib_fold + img + '.txt' for img in image]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


    def __getitem__(self, item):
        data_idx = self.id_list[item]
        unit_box2d = self.box2d_list[item]
        box3d_gt = self.box3d_gt_list[item]
        depth_gt = self.depth_min_list[item]
        depth_disturb = random.uniform(depth_gt - 3, depth_gt +3)
        left_box2d = self.left_box2d_list[item]
        left_img = self.dataset.get_image(data_idx)
        W, H = left_img.size
        right_img = self.dataset.get_right_image(data_idx)
        # calib = self.dataset.get_calibration(data_idx)
        # depth = self.dataset.get_depth(data_idx)
        # seg_mask = self.dataset.get_mask(data_idx)
        # seg_mask[seg_mask > 0] = 1
        # img_height, img_width, img_channel = left_img.shape
        calib_info = self.dataset.get_calibration(data_idx)
        xmin, ymin, xmax, ymax = unit_box2d
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        unit_box2d = np.array([xmin,ymin,xmax,ymax])
        # xmin, ymin = max(xmin, 0), max(ymin, 0)  # check range
        # xmax, ymax = min(xmax, W), min(ymax, H)  # check range
        # object_mask = seg_mask[int(ymin):int(ymax), int(xmin):int(xmax)]
        # object_depth = depth[int(ymin):int(ymax), int(xmin):int(xmax)]
        object_left_img = left_img.crop((xmin, ymin, xmax, ymax))
        object_right_img = right_img.crop((xmin, ymin, xmax, ymax))
        object_left_img = self.trans(object_left_img)
        object_right_img = self.trans(object_right_img)
        # depth_min = max(2, (depth_disturb - 10) / 224 * (int(xmax) - int(xmin)))
        # depth_max = (depth_disturb + 10) / 224 * (int(xmax) - int(xmin))
        depth_min = max(2.0, (depth_disturb-10) / 224 * (int(xmax) - int(xmin)))
        depth_max = min(40.0, (depth_disturb+10) / 224 * (int(xmax) - int(xmin)))
        # depth_min = max(2, (depth_disturb) / 224 * (int(xmax) - int(xmin)) - 10)
        # depth_max = min(40, (depth_disturb) / 224 * (int(xmax) - int(xmin)) + 10)
        # left_img = self.left_test[item]
        # right_img = self.right_test[item]
        # calib_info = read_calib_file(self.calib_test[item])
        if self.dynamic_bs:
            calib = calib_info.P[0, 0] * 0.54
            # calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * dynamic_baseline(calib_info)
        else:
            # calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * 0.54
            calib = calib_info.P[0, 0] * 0.54
        # imgL = Image.open(left_img).convert('RGB')
        # imgR = Image.open(right_img).convert('RGB')
        # imgL = self.trans(imgL)[None, :, :, :]
        # imgR = self.trans(imgR)[None, :, :, :]
        # # pad to (384, 1248)
        # B, C, H, W = imgL.shape
        # top_pad = 384 - H
        # right_pad = 1248 - W
        # imgL = F.pad(imgL, (0, right_pad, top_pad, 0), "constant", 0)
        # imgR = F.pad(imgR, (0, right_pad, top_pad, 0), "constant", 0)
        # filename = self.left_test[item].split('/')[-1][:-4]

        return object_left_img.float(), object_right_img.float(), calib.item(), np.array([depth_min,depth_max]),\
               data_idx, unit_box2d, box3d_gt,left_box2d

    def __len__(self):
        return len(self.id_list)

class SubmiteDataset1(object):
    def __init__(self, filepath, name, split, dynamic_bs=False, kitti2015=False):
        overwritten_data_path = os.path.join('/media/zd/2T/jcf/psmnet2_patch/',
                                             'kitti/frustum_caronly_psmnet_%s.pickle' % (name))
        with open(overwritten_data_path, 'rb') as fp:
            self.id_list = pickle.load(fp, encoding='iso-8859-1')
            self.box2d_list = pickle.load(fp, encoding='iso-8859-1')
            # self.box3d_gt_list = pickle.load(fp, encoding='iso-8859-1')
            # self.left_img_list = pickle.load(fp, encoding='iso-8859-1')
            # self.right_img_list = pickle.load(fp, encoding='iso-8859-1')
            # self.seg_mask_list = pickle.load(fp, encoding='iso-8859-1')
            # self.depth_list = pickle.load(fp, encoding='iso-8859-1')
            self.depth_min_list = pickle.load(fp, encoding='iso-8859-1')
            self.prob_list = pickle.load(fp, encoding='iso-8859-1')
            self.left_box2d_list = pickle.load(fp, encoding='iso-8859-1')
            self.frustum_angle_list = pickle.load(fp, encoding='iso-8859-1')

        self.dataset = kitti_object('/media/zd/2T/jcf/psmnet2_min/dataset/KITTI/object', 'training')

        self.dynamic_bs = dynamic_bs
        left_fold = 'image_2/'
        right_fold = 'image_3/'
        calib_fold = 'calib/'
        with open(split, 'r') as f:
            image = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        image = sorted(image)

        if kitti2015:
            self.left_test = [filepath + '/' + left_fold + img + '_10.png' for img in image]
            self.right_test = [filepath + '/' + right_fold + img + '_10.png' for img in image]
            self.calib_test = [filepath + '/' + calib_fold + img + '.txt' for img in image]
        else:
            self.left_test = [filepath + '/' + left_fold + img + '.png' for img in image]
            self.right_test = [filepath + '/' + right_fold + img + '.png' for img in image]
            self.calib_test = [filepath + '/' + calib_fold + img + '.txt' for img in image]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        self.dataset_helper = Kitti_Config()

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it is shifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + self.frustum_angle_list[index]

    def patchnet_pre(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)
        cls_type = 'Car'
        assert(cls_type in ['Car', 'Pedestrian', 'Cyclist'])
        one_hot_vec = np.zeros((3), dtype=np.float32)
        one_hot_vec[self.dataset_helper.type2onehot[cls_type]] = 1


        return rot_angle, one_hot_vec

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

    def __getitem__(self, item):
        data_idx = self.id_list[item]
        unit_box2d= self.box2d_list[item]
        left_box2d  = self.left_box2d_list[item]
        depth_gt = self.depth_min_list[item]
        # depth_disturb = random.uniform(depth_gt - 3, depth_gt + 3)
        depth_disturb = depth_gt
        # box3d_gt = self.box3d_gt_list[item]
        prob = self.prob_list[item]
        # depth_min = self.depth_min_list[index]
        left_img = self.dataset.get_image(data_idx)
        W, H = left_img.size
        right_img = self.dataset.get_right_image(data_idx)
        # calib = self.dataset.get_calibration(data_idx)
        depth = np.array(self.dataset.get_depth(data_idx))/256.0

        pad_h = (0, 384 - depth.shape[0])
        pad_w = (0, 1248 - depth.shape[1])

        pad_width = ((0, 0),pad_h, pad_w)
        depth = np.pad(depth[None], pad_width=pad_width,
                     mode='constant',
                     constant_values=0)
        depth = torch.from_numpy(depth)
        
        seg_mask = self.dataset.get_mask(data_idx)
        seg_mask = (seg_mask>0).astype(float)
        pad_h = (0, 384 - seg_mask.shape[0])
        pad_w = (0, 1248 - seg_mask.shape[1])

        pad_width = ((0, 0), pad_h, pad_w)
        seg_mask = np.pad(seg_mask[None], pad_width=pad_width,
                       mode='constant',
                       constant_values=0)
        seg_mask = torch.from_numpy(seg_mask)
        # seg_mask = self.dataset.get_mask(data_idx)
        # seg_mask[seg_mask > 0] = 1
        # img_height, img_width, img_channel = left_img.shape
        calib_info = self.dataset.get_calibration(data_idx)
        xmin, ymin, xmax, ymax = unit_box2d
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        unit_box2d = np.array([xmin,ymin,xmax,ymax])

        xmin1, ymin1, xmax1, ymax1 = left_box2d
        xmin1 = max(xmin1, xmin)
        ymin1, xmax1, ymax1 = ymin, xmax, ymax
        xmin1 = int(xmin1)
        ymin1 = int(ymin1)
        xmax1 = int(xmax1)
        ymax1 = int(ymax1)
        left_box2d = np.array([xmin1, ymin1, xmax1, ymax1])

        # xmin, ymin = max(xmin, 0), max(ymin, 0)  # check range
        # xmax, ymax = min(xmax, W), min(ymax, H)  # check range
        # object_mask = seg_mask[int(ymin):int(ymax), int(xmin):int(xmax)]
        # object_depth = depth[int(ymin):int(ymax), int(xmin):int(xmax)]
        # object_depth = torch.from_numpy(object_depth).unsqueeze(0)
        object_left_img = left_img.crop((xmin, ymin, xmax, ymax))
        object_right_img = right_img.crop((xmin, ymin, xmax, ymax))
        object_left_img = self.trans(object_left_img)
        object_right_img = self.trans(object_right_img)
        # depth_min = max(2, (depth_disturb - 10) / 224 * (int(xmax) - int(xmin)))
        # depth_max = (depth_disturb + 10) / 224 * (int(xmax) - int(xmin))
        depth_min = max(2.0, (depth_disturb-10) / 224 * (int(xmax) - int(xmin)))
        depth_max = min(40.0, (depth_disturb+10) / 224 * (int(xmax) - int(xmin)))
        # depth_min = max(2, (depth_disturb) / 224 * (int(xmax) - int(xmin))-10)
        # depth_max = min(40, (depth_disturb) / 224 * (int(xmax) - int(xmin))+10)
        # left_img = self.left_test[item]
        # right_img = self.right_test[item]
        # calib_info = read_calib_file(self.calib_test[item])
        if self.dynamic_bs:
            calib = calib_info.P[0, 0] * 0.54
            # calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * dynamic_baseline(calib_info)
        else:
            # calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * 0.54
            calib = calib_info.P[0, 0] * 0.54

        box_label = {}
        box_label['rot_angle'], box_label['one_hot_vec'] = self.patchnet_pre(item)
        # imgL = Image.open(left_img).convert('RGB')
        # imgR = Image.open(right_img).convert('RGB')
        # imgL = self.trans(imgL)[None, :, :, :]
        # imgR = self.trans(imgR)[None, :, :, :]
        # # pad to (384, 1248)
        # B, C, H, W = imgL.shape
        # top_pad = 384 - H
        # right_pad = 1248 - W
        # imgL = F.pad(imgL, (0, right_pad, top_pad, 0), "constant", 0)
        # imgR = F.pad(imgR, (0, right_pad, top_pad, 0), "constant", 0)
        # filename = self.left_test[item].split('/')[-1][:-4]

        return object_left_img.float(), object_right_img.float(), calib.item(), np.array([depth_min,depth_max]), \
               data_idx, unit_box2d, prob,np.array(left_box2d), calib_info.P, box_label


    def __len__(self):
        return len(self.id_list)
