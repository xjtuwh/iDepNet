import os
import shutil
import time

import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

import disp_models
import logger
import models
import utils_func
from dataloader import KITTILoader3D
from dataloader import KITTILoader_dataset3d
from dataloader import SceneFlowLoader
from dataloader import listflowfile
import pickle as pickle
from torch.autograd import Variable
from lib.losses.patchnet_loss import get_loss as patchnet_loss
from lib.utils.fpointnet_utils import write_detection_results

parser = configargparse.ArgParser(description='PSMNet')
parser.add('-c', '--config', required=True,
           is_config_file=True, help='config file')

parser.add_argument('--save_path', type=str, default='',
                    help='path to save the log, tensorbaord and checkpoint')
# network
parser.add_argument('--data_type', default='depth', choices=['disparity', 'depth'],
                    help='the network can predict either disparity or depth')
parser.add_argument('--arch', default='SDNet', choices=['SDNet', 'PSMNet'],
                    help='Model Name, default: SDNet.')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity, the range of the disparity cost volume: [0, maxdisp-1]')
parser.add_argument('--down', type=float, default=2,
                    help='reduce x times resolution when build the depth cost volume')
parser.add_argument('--maxdepth', type=int, default=40,
                    help='the range of the depth cost volume: [1, maxdepth]')
# dataset
parser.add_argument('--kitti2015', action='store_true',
                    help='If false, use 3d kitti dataset. If true, use kitti stereo 2015, default: False')
parser.add_argument('--dataset', default='kitti', choices=['sceneflow', 'kitti'],
                    help='train with sceneflow or kitti')
parser.add_argument('--datapath', default='',
                    help='root folder of the dataset')
parser.add_argument('--split_train', default='/media/zd/2T/kitti/KITTI/object/train.txt',
                    help='data splitting file for training')
parser.add_argument('--split_val', default='/media/zd/2T/kitti/KITTI/object/val.txt',
                    help='data splitting file for validation')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of training epochs')
parser.add_argument('--btrain', type=int, default=12,
                    help='training batch size')
parser.add_argument('--bval', type=int, default=4,
                    help='validation batch size')
parser.add_argument('--workers', type=int, default=4,
                    help='number of dataset workers')
# learning rate
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--lr_stepsize', nargs='+', type=int, default=[200],
                    help='drop lr in each step')
parser.add_argument('--lr_gamma', default=0.1, type=float,
                    help='gamma of the learning rate scheduler')
# resume
parser.add_argument('--resume', default=None,
                    help='path to a checkpoint')
parser.add_argument('--pretrain', default=None,
                    help='path to pretrained model')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch')
# evaluate
parser.add_argument('--evaluate', action='store_true',
                    help='do evaluation')
parser.add_argument('--calib_value', type=float, default=1017,
                    help='manually define focal length. (sceneflow does not have configuration)')
parser.add_argument('--dynamic_bs', action='store_true',
                    help='If true, dynamically calculate baseline from calibration file. If false, use 0.54')
parser.add_argument('--eval_interval', type=int, default=100,
                    help='evaluate model every n epochs')
parser.add_argument('--checkpoint_interval', type=int, default=10,
                    help='save checkpoint every n epoch.')
parser.add_argument('--generate_depth_map', action='store_true',
                    help='if true, generate depth maps and save the in save_path/depth_maps/{data_tag}/')
parser.add_argument('--data_list', default=None,
                    help='generate depth maps for all the data in this list')
parser.add_argument('--data_tag', default=None,
                    help='the suffix of the depth maps folder')
args = parser.parse_args()
best_RMSE = 1e10


def img_to_rect(u, v, depth_rect, P2):
    """
    :param u: (N)
    :param v: (N)
    :param depth_rect: (N)
    :return:
    """
    x = ((u - P2[0,2]) * depth_rect) / P2[0,0]
    y = ((v - P2[1,2]) * depth_rect) / P2[1,1]
    pts_rect = torch.stack((x, y, depth_rect), axis=1)
    return pts_rect

def rotate_pc_along_y(pc, rot_angle):
    """
    params pc: (N, 3+C), (N, 3) is in the rectified camera coordinate
    params rot_angle: rad scalar
    Output pc: updated pc with XYZ rotated
    """
    cosval = torch.cos(rot_angle)
    sinval = torch.sin(rot_angle)
    rotmat = torch.tensor([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = torch.mm(pc[:, [0, 2]], rotmat.transpose(0,1).cuda())
    return pc

def rotato_patch_to_center(patch, angle):
    # Use np.copy to avoid corrupting original data
    h, w, c = patch.size()
    xyz = torch.clone(patch).reshape(-1, 3)
    xyz_updated = rotate_pc_along_y(xyz, angle)
    patch_updated = xyz_updated.reshape(h, w, c)
    return patch_updated

def main():
    global best_RMSE

    # set logger
    log = logger.setup_logger(os.path.join(args.save_path, 'training.log'))
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    # set tensorboard
    writer = SummaryWriter(args.save_path + '/tensorboardx')

    # Data Loader
    if args.generate_depth_map:
        TrainImgLoader = None
        import dataloader.KITTI_submission_loader  as KITTI_submission_loader
        TestImgLoader = torch.utils.data.DataLoader(
            KITTI_submission_loader.SubmiteDataset(args.datapath, 'train1', args.data_list, args.dynamic_bs),
            batch_size=args.bval, shuffle=False, num_workers=args.workers, drop_last=False)
        TestImgLoader1 = torch.utils.data.DataLoader(
            KITTI_submission_loader.SubmiteDataset1(args.datapath, 'detection', args.data_list, args.dynamic_bs),
            batch_size=args.bval, shuffle=False, num_workers=args.workers, drop_last=False)

    elif args.dataset == 'kitti':
        train_data, val_data = KITTILoader3D.dataloader(args.datapath, args.split_train, args.split_val,
                                                        kitti2015=args.kitti2015)
        TrainImgLoader = torch.utils.data.DataLoader(
            KITTILoader_dataset3d.myImageFloder(train_data, True, 'train', kitti2015=args.kitti2015, dynamic_bs=args.dynamic_bs),
            batch_size=args.btrain, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
        # TestImgLoader = torch.utils.data.DataLoader(
        #     KITTILoader_dataset3d.myImageFloder(val_data, False, 'val',kitti2015=args.kitti2015, dynamic_bs=args.dynamic_bs),
        #     batch_size=args.bval, shuffle=False, num_workers=args.workers, drop_last=True, pin_memory=True)
        TestImgLoader = TrainImgLoader
    else:
        train_data, val_data = listflowfile.dataloader(args.datapath)
        TrainImgLoader = torch.utils.data.DataLoader(
            SceneFlowLoader.myImageFloder(train_data, True, calib=args.calib_value),
            batch_size=args.btrain, shuffle=True, num_workers=args.workers, drop_last=False)
        TestImgLoader = torch.utils.data.DataLoader(
            SceneFlowLoader.myImageFloder(val_data, False, calib=args.calib_value),
            batch_size=args.bval, shuffle=False, num_workers=args.workers, drop_last=False)

    # Load Model
    if args.data_type == 'disparity':
        model = disp_models.__dict__[args.arch](maxdisp=args.maxdisp)
    elif args.data_type == 'depth':
        model = models.__dict__[args.arch](maxdepth=args.maxdepth, maxdisp=args.maxdisp, down=args.down)
    else:
        log.info('Model is not implemented')
        assert False

    # Number of parameters
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    model = nn.DataParallel(model).cuda()
    torch.backends.cudnn.benchmark = True

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = MultiStepLR(optimizer, milestones=args.lr_stepsize, gamma=args.lr_gamma)

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            log.info("=> loading pretrain '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            log.info('[Attention]: Can not find checkpoint {}'.format(args.pretrain))

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'],strict = False)
            # args.start_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # best_RMSE = checkpoint['best_RMSE']
            # scheduler.load_state_dict(checkpoint['scheduler'],strict = False)
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info('[Attention]: Can not find checkpoint {}'.format(args.resume))

    if args.generate_depth_map:
        os.makedirs(args.save_path + '/depth_maps+/' + args.data_tag, exist_ok=True)
        # id_list = []  # int number
        # box2d_list = []
        # box3d_gt_list = []
        # left_box2d_list = []
        # tqdm_eval_loader = tqdm(TestImgLoader, total=len(TestImgLoader))
        # num = 0
        # for batch_idx, (imgL_crop, imgR_crop, calib, depth_min, idx, unit_box2d,box3d_gt,left_box2d) in enumerate(tqdm_eval_loader):
        #     pred_disp = inference(imgL_crop, imgR_crop, calib, depth_min, model)
        #     unit_box2d = unit_box2d.detach().cpu().numpy()
        #     idx = idx.detach().cpu().numpy()
        #     box3d_gt = box3d_gt.detach().cpu().numpy()
        #     left_box2d = left_box2d.detach().cpu().numpy()
        #     for i in range(unit_box2d.shape[0]):
        #         id_list.append(idx[i])
        #         box2d_list.append(unit_box2d[i])
        #         box3d_gt_list.append((box3d_gt[i]))
        #         left_box2d_list.append(left_box2d[i])
        #         obj_disp = torch.from_numpy(pred_disp[i]).float()  # H * W * C
        #         obj_disp = obj_disp.unsqueeze(0)
        #         obj_disp = obj_disp.unsqueeze(0)
        #         box_w, box_h = int(unit_box2d[i][2]-unit_box2d[i][0]), int(unit_box2d[i][3]-unit_box2d[i][1])
        #         disp = F.interpolate(obj_disp, (box_h, box_w), mode='bilinear', align_corners=True).squeeze().numpy()
        #         disp=disp/disp.shape[1]*224
        #         depth_filename = os.path.join('/media/zd/2T/jcf/psmnet2_min/kitti/depth_map', '%06d' % (num+i))
        #         np.save(depth_filename, disp)
        #     num = num+4
        # with open('/media/zd/2T/jcf/psmnet2_min/kitti/caronly_psmnet_output_train.pickle', 'wb') as fp:
        #     pickle.dump(id_list, fp)
        #     pickle.dump(box2d_list, fp)
        #     pickle.dump(box3d_gt_list, fp)
        #     pickle.dump(left_box2d_list, fp)

        id_list1 = []  # int number
        box2d_list1 = []
        depth_list1 = []
        prob_list = []
        left_box2d_list = []

        center_list = []
        heading_cls_list = []
        heading_res_list = []
        size_cls_list = []
        size_res_list = []
        rot_angle_list = []
        score_list = []
        id_list = []
        type_list = []
        box2d_list = []
        tqdm_eval_loader1 = tqdm(TestImgLoader1, total=len(TestImgLoader1))
        model.eval()
        torch.set_grad_enabled(False)
        for batch_idx, (imgL_crop, imgR_crop, calib, depth_min, idx, unit_box2d, prob,left_box2d,P2,box_label) in enumerate(tqdm_eval_loader1):
            # if batch_idx==2794:
            #     break
            for key in box_label.keys():
                box_label[key] = box_label[key].cuda()
            P2 = P2.cuda()
            pred_disp = inference(imgL_crop, imgR_crop, calib, depth_min,model)

            patch_tensor = Variable(torch.cuda.FloatTensor(pred_disp.shape[0], 3, 32, 32).zero_())

            y = torch.range(0, 1247).cuda()
            x = torch.range(0, 383).cuda()
            xx, yy = torch.meshgrid(x, y)
            uv = torch.stack((yy, xx), dim=2)
            for i in range(unit_box2d.shape[0]):
                id_list1.append(idx[i])
                box2d_list1.append(unit_box2d[i])
                obj_disp = torch.from_numpy(pred_disp[i]).float()  # H * W * C
                obj_disp = obj_disp.unsqueeze(0)
                obj_disp = obj_disp.unsqueeze(0)
                box_w, box_h = int(unit_box2d[i][2] - unit_box2d[i][0]), int(unit_box2d[i][3] - unit_box2d[i][1])
                disp = F.interpolate(obj_disp, (box_h, box_w), mode='bilinear', align_corners=True).squeeze()
                disp = disp / disp.shape[1] * 224
                left_x1 = int(left_box2d[i][0] - unit_box2d[i][0])
                disp = disp[:,left_x1:]
                height, width = disp.size()
                uvdepth = Variable(torch.cuda.FloatTensor(height, width, 3).zero_())
                uvdepth[:, :, 0:2] = uv[left_box2d[i, 1]:left_box2d[i, 3], left_box2d[i, 0]:left_box2d[i, 2]]
                uvdepth[:, :, 2] = disp
                uvdepth = uvdepth.view(-1, 3)
                patch = img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2], P2[i])
                patch = patch.reshape(height, width, 3)
                patch[:, :, 0:3] = rotato_patch_to_center(patch[:, :, 0:3], box_label['rot_angle'][i])
                patch = patch.unsqueeze(0)  # 1 * H * W * C
                patch = patch.transpose(2, 3).transpose(1, 2)  # 1 * H * W * C -> 1 * H * C * W  ->  1 * C * H * W
                patch_tensor[i] = F.interpolate(patch, (32, 32), mode='bilinear', align_corners=True).squeeze(0)
            outputs = model.module.patchnet(patch_tensor, box_label['one_hot_vec'])

            outputs['center'] = outputs['center'].cpu().numpy()
            outputs['heading_scores'] = outputs['heading_scores'].cpu().numpy()
            outputs['heading_residuals'] = outputs['heading_residuals'].cpu().numpy()
            outputs['size_scores'] = outputs['size_scores'].cpu().numpy()
            outputs['size_residuals'] = outputs['size_residuals'].cpu().numpy()

            rot_angle = box_label['rot_angle'].cpu().numpy()
            rgb_prob = prob.numpy()
            id = idx.numpy()
            box2d = left_box2d.numpy()
            batch_size = patch_tensor.shape[0]
            for i in range(batch_size):
                center_list.append(outputs['center'][i, :])
                heading_cls = np.argmax(outputs['heading_scores'][i, :])
                heading_cls_list.append(heading_cls)
                heading_res = outputs['heading_residuals'][i, heading_cls]
                heading_res_list.append(heading_res)
                size_cls = np.argmax(outputs['size_scores'][i, :])
                size_cls_list.append(size_cls)
                size_res = outputs['size_residuals'][i][size_cls]
                size_res_list.append(size_res)
                rot_angle_list.append(rot_angle[i])
                score_list.append(rgb_prob[i])  # 2D RGB detection score
                id_list.append(id[i])
                # type_list.append(type[i])
                type_list.append('Car')
                box2d_list.append(box2d[i])
        print('Write detection results for KITTI evaluation')
        result_dir = './output'
        os.makedirs('./output', exist_ok=True)
        write_detection_results(result_dir=result_dir,
                                id_list=id_list,
                                type_list=type_list,
                                box2d_list=box2d_list,
                                center_list=center_list,
                                heading_cls_list=heading_cls_list,
                                heading_res_list=heading_res_list,
                                size_cls_list=size_cls_list,
                                size_res_list=size_res_list,
                                rot_angle_list=rot_angle_list,
                                score_list=score_list)

        # with open('/media/zd/2T/jcf/psmnet2_min/kitti/caronly_psmnet_output_detection.pickle', 'wb') as fp:
        #     pickle.dump(id_list1, fp)
        #     pickle.dump(box2d_list1, fp)
        #     pickle.dump(depth_list1, fp)
        #     pickle.dump(prob_list, fp)
        #     pickle.dump(left_box2d_list, fp)
        import sys
        sys.exit()

    # evaluation
    if args.evaluate:
        evaluate_metric = utils_func.Metric()
        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, calib) in enumerate(TestImgLoader):
            start_time = time.time()
            test(imgL_crop, imgR_crop, disp_crop_L, calib, evaluate_metric, model)

            log.info(evaluate_metric.print(batch_idx, 'EVALUATE') + ' Time:{:.3f}'.format(time.time() - start_time))
        import sys
        sys.exit()

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        ## training ##
        # train_metric = utils_func.Metric()
        # tqdm_train_loader = tqdm(TrainImgLoader, total=len(TrainImgLoader))
        # for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, seg_mask, calib, depth_min,unit_box2d,left_box2d,P2,\
        #                 box_label) in enumerate(tqdm_train_loader):
        #     # start_time = time.time()
        #     patch_loss, loss= train(imgL_crop, imgR_crop, disp_crop_L, seg_mask, calib, depth_min, unit_box2d,left_box2d,P2, \
        #           box_label,train_metric, optimizer, model)
        #
        #     # log.info(train_metric.print(batch_idx, 'TRAIN') + ' Time:{:.3f}'.format(time.time() - start_time))
        #     if batch_idx%100==0:
        #         print("epoch: %d  patch_loss: %f,total_loss:%f"  %(epoch,patch_loss,loss))
        # log.info(train_metric.print(0, 'TRAIN Epoch' + str(epoch)))
        # train_metric.tensorboard(writer, epoch, token='TRAIN')

        ## testing ##
        is_best = False
        if (epoch % args.eval_interval) == 100:
            test_metric = utils_func.Metric()
            tqdm_test_loader = tqdm(TestImgLoader, total=len(TestImgLoader))
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, seg_mask, calib, depth_min) in enumerate(tqdm_test_loader):
                # start_time = time.time()
                test(imgL_crop, imgR_crop, disp_crop_L, seg_mask, calib, depth_min, test_metric, model)
                # log.info(test_metric.print(batch_idx, 'TEST') + ' Time:{:.3f}'.format(time.time() - start_time))
            log.info(test_metric.print(0, 'TEST Epoch' + str(epoch)))
            test_metric.tensorboard(writer, epoch, token='TEST')

            # SAVE
            is_best = test_metric.RMSELIs.avg < best_RMSE
            best_RMSE = min(test_metric.RMSELIs.avg, best_RMSE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_RMSE': best_RMSE,
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, folder=args.save_path)
    # lw.done()


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar', folder='result/default'):
    torch.save(state, folder + '/' + filename)
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')
    if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
        shutil.copyfile(folder + '/' + filename, folder + '/checkpoint_{}.pth.tar'.format(epoch + 1))

def compute_seg_loss(output, target):
    output = output.permute(0,2,3,1)
    # target = target.unsqueeze(3)
    return F.cross_entropy(output.reshape(8*192*640,2), target.reshape(8*192*640).long())

def train(imgL, imgR, depth, seg_mask, calib, depth_min, unit_box2d,left_box2d, P2, box_label,metric_log, optimizer, model):
    model.train()
    calib = calib.float()
    # depth_min = depth_min.float()

    imgL, imgR, depth, seg_mask, calib, depth_min,unit_box2d,left_box2d,P2 = imgL.cuda(), imgR.cuda(), \
                                                   depth.cuda(),seg_mask.cuda(),calib.cuda(),depth_min.cuda(),\
                                                   unit_box2d.cuda(),left_box2d.cuda(),P2.cuda()

    for key in box_label.keys():
        box_label[key] = box_label[key].cuda()
    # ---------
    mask = (depth >= 1) * (depth <= 80)*(seg_mask>0.9)
    mask.detach_()
    # ----

    optimizer.zero_grad()

    output1, output2, output3= model(imgL, imgR, calib, depth_min)
    output1 = torch.squeeze(output1, 1)
    output2 = torch.squeeze(output2, 1)
    output3 = torch.squeeze(output3, 1)
    if args.data_type == 'disparity':
        output1 = disp2depth(output1, calib)
        output2 = disp2depth(output2, calib)
        output3 = disp2depth(output3, calib)

    obj_disp = output3.unsqueeze(1)
    patch_tensor = Variable(torch.cuda.FloatTensor(output3.size()[0],3, 32, 32).zero_())

    x = torch.range(0,383).cuda()
    y = torch.range(0, 1247).cuda()
    xx,yy = torch.meshgrid(x,y)
    uv = torch.stack((yy,xx),dim=2)
    for i in range(obj_disp.size()[0]):
        box_w, box_h = unit_box2d[i,2] - unit_box2d[i,0], unit_box2d[i,3] - unit_box2d[i,1]
        disp = F.interpolate(obj_disp[i:i+1], (box_h, box_w), mode='bilinear', align_corners=True).squeeze()
        disp = disp / disp.size()[1] * 224
        left_x1 = left_box2d[i][0] - unit_box2d[i][0]
        disp = disp[:, left_x1:]
        height, width = disp.size()
        uvdepth = Variable(torch.cuda.FloatTensor(height, width, 3).zero_())
        uvdepth[:,:,0:2]= uv[left_box2d[i,1]:left_box2d[i,3],left_box2d[i,0]:left_box2d[i,2]]
        uvdepth[:, :, 2] = disp
        uvdepth = uvdepth.view(-1, 3)
        patch = img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2],P2[i])
        patch = patch.reshape(height, width, 3)
        patch[:, :, 0:3] = rotato_patch_to_center(patch[:, :, 0:3], box_label['rot_angle'][i])
        patch = patch.unsqueeze(0)  # 1 * H * W * C
        patch = patch.transpose(2, 3).transpose(1, 2)  # 1 * H * W * C -> 1 * H * C * W  ->  1 * C * H * W
        patch_tensor[i] = F.interpolate(patch, (32,32), mode='bilinear', align_corners=True).squeeze(0)

    output_dict = model.module.patchnet(patch_tensor, box_label['one_hot_vec'])
    patch_loss = patchnet_loss(box_label['center'], box_label['angle_class'], box_label['angle_residual'], \
                         box_label['size_class'], box_label['size_residual'],model.module.patchnet.num_heading_bin, \
                         model.module.patchnet.num_size_cluster, model.module.patchnet.mean_size_arr, output_dict)

    loss = 0.5 * F.smooth_l1_loss(output1[mask], depth[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
        output2[mask], depth[mask], size_average=True) + F.smooth_l1_loss(output3[mask], depth[mask],
                                                                          size_average=True)+0.01*patch_loss
    # patch_loss = torch.tensor([0.0])
    metric_log.calculate(depth, output3, loss=loss.item())
    loss.backward()
    optimizer.step()
    return patch_loss,loss


def inference(imgL, imgR, calib, depth_min, model):
    model.eval()
    imgL, imgR, calib, depth_min= imgL.cuda(), imgR.cuda(), calib.float().cuda(),depth_min.cuda()

    with torch.no_grad():
        output = model(imgL, imgR, calib, depth_min)
    if args.data_type == 'disparity':
        output = disp2depth(output, calib)
    pred_disp = output.data.cpu().numpy()
    # pred_disp =output
    return pred_disp


def test(imgL, imgR,  depth, seg_mask, calib, depth_min, metric_log, model):
    model.eval()
    calib = calib.float()
    # imgL, imgR, calib, depth = imgL.cuda(), imgR.cuda(), calib.cuda(), depth.cuda()
    imgL, imgR, depth, seg_mask, calib, depth_min = imgL.cuda(), imgR.cuda(), depth.cuda(), \
                                                    seg_mask.cuda(), calib.cuda(), depth_min.cuda()

    mask = (depth >= 1) * (depth <= 80)*(seg_mask>0.5)
    mask.detach_()
    with torch.no_grad():
        output3 = model(imgL, imgR, calib, depth_min)
        output3 = torch.squeeze(output3, 1)

        if args.data_type == 'disparity':
            output3 = disp2depth(output3, calib)
        loss = F.smooth_l1_loss(output3[mask], depth[mask], size_average=True)

        metric_log.calculate(depth, output3, loss=loss.item())

    torch.cuda.empty_cache()
    return


def disp2depth(disp, calib):
    depth = calib[:, None, None] / disp.clamp(min=1e-8)
    return depth


if __name__ == '__main__':
    main()
