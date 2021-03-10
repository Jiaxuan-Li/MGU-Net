#!/home/user1/anaconda3/envs/env_py37/bin/python
# -*- coding: utf-8 -*-
# @Author:Jiaxuan Li
##### System library #####
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import os.path as osp
from os.path import exists
import argparse
import json
import logging
import time
import numpy as np
import shutil
import random
import copy
##### pytorch library #####
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
##### My own library #####
import data.seg_transforms as dt
from data.seg_dataset import segList
from utils.logger import Logger
from models.net_builder import net_builder
from utils.loss import loss_builder1,loss_builder2
from utils.utils import adjust_learning_rate
from utils.utils import AverageMeter, save_model
from utils.utils import compute_dice,compute_pa,compute_avg_score_batch,compute_single_avg_score
from utils.vis import vis_result

# logger vis
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger_vis = logging.getLogger(__name__)
logger_vis.setLevel(logging.DEBUG)

# training process
def train(args,train_loader, model, criterion2, optimizer,epoch,print_freq=10):
   # set the AverageMeter 
    batch_time = AverageMeter()
    losses = AverageMeter()
    dice = AverageMeter()
    Dice_1, Dice_2, Dice_3, Dice_4, Dice_5, Dice_6, Dice_7, Dice_8, Dice_9, Dice_10 = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # variable

        input_var = Variable(input).cuda()
        target_var_seg = Variable(target).cuda()
        input_var1 = copy.deepcopy(input_var)
        # forward
        output_seg = model(input_var1)
        # calculate loss
        loss_2_1 = criterion2[0](output_seg, target_var_seg)
        loss_2_2 = criterion2[1](output_seg, target_var_seg)
        loss_2= loss_2_1 + loss_2_2     # loss from the two-stage network       
        loss = loss_2
        #print(loss)
        losses.update(loss.data, input.size(0))
        # calculate dice score for segmentation 
        _, pred_seg = torch.max(output_seg, 1)
        pred_seg = pred_seg.cpu().data.numpy()
        label_seg = target_var_seg.cpu().data.numpy()
        ret_d = compute_dice(label_seg, pred_seg)
        dice_score = compute_single_avg_score(ret_d)
        # update dice score
        dice.update(dice_score)
        Dice_1.update(ret_d[1])
        Dice_2.update(ret_d[2])
        Dice_3.update(ret_d[3])
        Dice_4.update(ret_d[4])
        Dice_5.update(ret_d[5])
        Dice_6.update(ret_d[6])
        Dice_7.update(ret_d[7])
        Dice_8.update(ret_d[8])
        Dice_9.update(ret_d[9])
        Dice_10.update(ret_d[10])
        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # logger vis
        if i % print_freq == 0:
            logger_vis.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Dice {dice.val:.4f} ({dice.avg:.4f})\t'
                        'Dice_1 {dice_1.val:.4f} ({dice_1.avg:.4f})\t'
                        'Dice_2 {dice_2.val:.4f} ({dice_2.avg:.4f})\t'
                        'Dice_3 {dice_3.val:.4f} ({dice_3.avg:.4f})\t'
                        'Dice_4 {dice_4.val:.4f} ({dice_4.avg:.4f})\t'
                        'Dice_5 {dice_5.val:.4f} ({dice_5.avg:.4f})\t'
                        'Dice_6 {dice_6.val:.4f} ({dice_6.avg:.4f})\t'
                        'Dice_7 {dice_7.val:.4f} ({dice_7.avg:.4f})\t'
                        'Dice_8 {dice_8.val:.4f} ({dice_8.avg:.4f})\t'
                        'Dice_9 {dice_9.val:.4f} ({dice_9.avg:.4f})\t'
                        'Dice_10 {dice_10.val:.4f} ({dice_10.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,dice = dice,dice_1=Dice_1,dice_2=Dice_2,dice_3=Dice_3,dice_4=Dice_4,dice_5=Dice_5,dice_6=Dice_6,dice_7=Dice_7,dice_8=Dice_8,dice_9=Dice_9,dice_10=Dice_10))
            print('Loss :',loss.cpu().data.numpy())           
    return losses.avg,dice.avg,Dice_1.avg,Dice_2.avg,Dice_3.avg,Dice_4.avg,Dice_5.avg,Dice_6.avg,Dice_7.avg,Dice_8.avg,Dice_9.avg,Dice_10.avg

# evaluation process
def eval(phase, args, eval_data_loader, model,result_path = None, logger = None):
    # set the AverageMeter 
    batch_time = AverageMeter()
    dice = AverageMeter()
    mpa = AverageMeter()
    Dice_1, Dice_2, Dice_3, Dice_4, Dice_5, Dice_6, Dice_7, Dice_8, Dice_9, Dice_10 = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    pa_1, pa_2, pa_3, pa_4, pa_5, pa_6, pa_7, pa_8, pa_9, pa_10 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    dice_list, mpa_list = [], []
    ret_dice, ret_pa = [], []
    # switch to eval mode
    model.eval()
    end = time.time()
    pred_seg_batch = []
    label_seg_batch = []
    for iter, (image, label, imt, imn) in enumerate(eval_data_loader):
        with torch.no_grad():
            image_var = Variable(image).cuda()
            # model forward
            output_seg = model(image_var)
            _, pred_seg = torch.max(output_seg, 1)
            # save visualized result
            pred_seg = pred_seg.cpu().data.numpy().astype('uint8')
            if phase == 'eval' or phase == 'test':
                imt = (imt.squeeze().numpy()).astype('uint8')
                ant = label.numpy().astype('uint8')
                save_dir = osp.join(result_path, 'vis')
                if not exists(save_dir): os.makedirs(save_dir)
                if not exists(save_dir+'/label'):os.makedirs(save_dir+'/label')
                if not exists(save_dir + '/pred'): os.makedirs(save_dir + '/pred')
                vis_result(imn, imt, ant, pred_seg, save_dir)
                print('Saved visualized results!')
            # calculate dice and pa score for segmentation
            label_seg = label.numpy().astype('uint8')
            pred_seg_batch.append(pred_seg)
            label_seg_batch.append(label_seg)
            ret_d = compute_dice(label_seg, pred_seg)
            ret_p = compute_pa(label_seg, pred_seg)
            ret_dice.append(ret_d)
            ret_pa.append(ret_p)
            dice_score = compute_single_avg_score(ret_d)
            mpa_score = compute_single_avg_score(ret_p)
            dice_list.append(dice_score)
            # update dice and pa score
            dice.update(dice_score)
            Dice_1.update(ret_d[1])
            Dice_2.update(ret_d[2])
            Dice_3.update(ret_d[3])
            Dice_4.update(ret_d[4])
            Dice_5.update(ret_d[5])
            Dice_6.update(ret_d[6])
            Dice_7.update(ret_d[7])
            Dice_8.update(ret_d[8])
            Dice_9.update(ret_d[9])
            Dice_10.update(ret_d[10])
            mpa_list.append(mpa_score)
            mpa.update(mpa_score)
            pa_1.update(ret_p[1])
            pa_2.update(ret_p[2])
            pa_3.update(ret_p[3])
            pa_4.update(ret_p[4])
            pa_5.update(ret_p[5])
            pa_6.update(ret_p[6])
            pa_7.update(ret_p[7])
            pa_8.update(ret_p[8])
            pa_9.update(ret_p[9])
            pa_10.update(ret_p[10])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger_vis.info('{0}: [{1}/{2}]\t'
                        'ID {id}\t'
                        'Dice {dice.val:.4f}\t'
                        'Dice_1 {dice_1.val:.4f}\t'
                        'Dice_2 {dice_2.val:.4f}\t'
                        'Dice_3 {dice_3.val:.4f}\t'
                        'Dice_4 {dice_4.val:.4f}\t'
                        'Dice_5 {dice_5.val:.4f}\t'
                        'Dice_6 {dice_6.val:.4f}\t'
                        'Dice_7 {dice_7.val:.4f}\t'
                        'Dice_8 {dice_8.val:.4f}\t'
                        'Dice_9 {dice_9.val:.4f}\t'
                        'Dice_10 {dice_10.val:.4f}\t'
                        'MPA {mpa.val:.4f}\t'
                        'PA_1 {pa_1.val:.4f}\t'
                        'PA_2 {pa_2.val:.4f}\t'
                        'PA_3 {pa_3.val:.4f}\t'
                        'PA_4 {pa_4.val:.4f}\t'
                        'PA_5 {pa_5.val:.4f}\t'
                        'PA_6 {pa_6.val:.4f}\t'
                        'PA_7 {pa_7.val:.4f}\t'
                        'PA_8 {pa_8.val:.4f}\t'
                        'PA_9 {pa_9.val:.4f}\t'
                        'PA_10 {pa_10.val:.4f}\t'
                        'Batch_time {batch_time.val:.3f}\t'
                        .format(phase.upper(), iter, len(eval_data_loader),id=imn[0].split('.')[0], dice=dice, dice_1=Dice_1, dice_2=Dice_2, dice_3=Dice_3,
                                dice_4=Dice_4, dice_5=Dice_5, dice_6=Dice_6, dice_7=Dice_7, dice_8=Dice_8,
                                dice_9=Dice_9, dice_10=Dice_10, mpa=mpa, pa_1=pa_1, pa_2=pa_2, pa_3=pa_3,
                                pa_4=pa_4, pa_5=pa_5, pa_6=pa_6, pa_7=pa_7, pa_8=pa_8,
                                pa_9=pa_9, pa_10=pa_10, batch_time=batch_time))
    # print final all dice and pa score 
    """
    pred_seg_batch = np.array(pred_seg_batch)
    label_seg_batch = np.array(label_seg_batch)
    ret_dice_batch = compute_dice(label_seg_batch, pred_seg_batch)
    ret_pa_batch = compute_pa(label_seg_batch, pred_seg_batch)
    final_dice_avg, final_dice_1, final_dice_2, final_dice_3, final_dice_4, final_dice_5, final_dice_6, final_dice_7, final_dice_8, final_dice_9, final_dice_10 = compute_avg_score_batch(ret_dice_batch)
    final_pa_avg, final_pa_1, final_pa_2, final_pa_3, final_pa_4, final_pa_5, final_pa_6, final_pa_7, final_pa_8, final_pa_9, final_pa_10 = compute_avg_score_batch(ret_pa_batch)
    """
    final_dice_avg, final_dice_1, final_dice_2, final_dice_3, final_dice_4, final_dice_5, final_dice_6, final_dice_7, final_dice_8, final_dice_9, final_dice_10 = dice.avg, Dice_1.avg, Dice_2.avg, Dice_3.avg, Dice_4.avg, Dice_5.avg, Dice_6.avg, Dice_7.avg, Dice_8.avg, Dice_9.avg, Dice_10.avg
    final_pa_avg, final_pa_1, final_pa_2, final_pa_3, final_pa_4, final_pa_5, final_pa_6, final_pa_7, final_pa_8, final_pa_9, final_pa_10 = mpa.avg, pa_1.avg, pa_2.avg, pa_3.avg, pa_4.avg, pa_5.avg, pa_6.avg, pa_7.avg, pa_8.avg, pa_9.avg, pa_10.avg
    print('######  Segmentation Result  ######')
    print('Final Dice_avg Score:{:.4f}'.format(final_dice_avg))
    print('Final Dice_1 Score:{:.4f}'.format(final_dice_1))
    print('Final Dice_2 Score:{:.4f}'.format(final_dice_2))
    print('Final Dice_3 Score:{:.4f}'.format(final_dice_3))
    print('Final Dice_4 Score:{:.4f}'.format(final_dice_4))
    print('Final Dice_5 Score:{:.4f}'.format(final_dice_5))
    print('Final Dice_6 Score:{:.4f}'.format(final_dice_6))
    print('Final Dice_7 Score:{:.4f}'.format(final_dice_7))
    print('Final Dice_8 Score:{:.4f}'.format(final_dice_8))
    print('Final Dice_9 Score:{:.4f}'.format(final_dice_9))
    print('Final Dice_10 Score:{:.4f}'.format(final_dice_10))
    print('Final PA_avg:{:.4f}'.format(final_pa_avg))
    print('Final PA_1 Score:{:.4f}'.format(final_pa_1))
    print('Final PA_2 Score:{:.4f}'.format(final_pa_2))
    print('Final PA_3 Score:{:.4f}'.format(final_pa_3))
    print('Final PA_4 Score:{:.4f}'.format(final_pa_4))
    print('Final PA_5 Score:{:.4f}'.format(final_pa_5))
    print('Final PA_6 Score:{:.4f}'.format(final_pa_6))
    print('Final PA_7 Score:{:.4f}'.format(final_pa_7))
    print('Final PA_8 Score:{:.4f}'.format(final_pa_8))
    print('Final PA_9 Score:{:.4f}'.format(final_pa_9))
    print('Final PA_10 Score:{:.4f}'.format(final_pa_10))
    if phase == 'eval' or phase == 'test':
        logger.append(
        [ final_dice_avg, final_dice_1, final_dice_2, final_dice_3, final_dice_4, final_dice_5, final_dice_6, final_dice_7, final_dice_8, final_dice_9,  final_dice_10,
        final_pa_avg, final_pa_1, final_pa_2, final_pa_3, final_pa_4, final_pa_5, final_pa_6, final_pa_7, final_pa_8, final_pa_9, final_pa_10])
    return final_dice_avg, final_dice_1, final_dice_2, final_dice_3, final_dice_4, final_dice_5, final_dice_6, final_dice_7, final_dice_8, final_dice_9, final_dice_10,dice_list


###### train ######
def train_seg(args,train_result_path,train_loader,eval_loader):
    # logger setting
    logger_train = Logger(osp.join(train_result_path,'dice_epoch.txt'), title='dice',resume=False)
    logger_train.set_names(['Epoch','Dice_Train','Dice_Val','Dice_1','Dice_11','Dice_2','Dice_22','Dice_3','Dice_33','Dice_4','Dice_44','Dice_5','Dice_55','Dice_6','Dice_66','Dice_7','Dice_77','Dice_8','Dice_88','Dice_9','Dice_99','Dice_10','Dice_1010',])
    # print hyperparameters
    for k, v in args.__dict__.items():
        print(k, ':', v)
    # load the network
    net = net_builder(args.name)
    model = torch.nn.DataParallel(net).cuda()
    print('#'*15,args.name,'#'*15)
    # define loss function
    #criterion1 = loss_builder1(args.loss)
    criterion2 = loss_builder2(args.loss)
    # set optimizer
    optimizer = torch.optim.Adam(net.parameters(), #Adam optimizer
                                    args.lr,
                                    betas=(0.9, 0.99),
                                    weight_decay=args.weight_decay)     
    cudnn.benchmark = True
    # main training
    best_dice = 0
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args,optimizer, epoch)
        logger_vis.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        loss,dice_train,dice_1,dice_2,dice_3,dice_4,dice_5,dice_6,dice_7,dice_8,dice_9,dice_10 = train(args,train_loader, model,criterion2, optimizer,epoch)
        # evaluate on validation set
        dice_val,dice_11,dice_22,dice_33,dice_44,dice_55,dice_66,dice_77,dice_88,dice_99,dice_1010,dice_list = eval('train', args, eval_loader, model)
        # save the best model
        is_best = dice_val > best_dice
        best_dice = max(dice_val, best_dice)
        model_dir = osp.join(train_result_path,'model')
        if not exists(model_dir):
            os.makedirs(model_dir)
        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'dice_epoch':dice_val,
            'best_dice': best_dice,
        }, is_best, model_dir)
        # logger 
        logger_train.append([epoch,dice_train,dice_val,dice_1,dice_11,dice_2,dice_22,dice_3,dice_33,dice_4,dice_44,dice_5,dice_55,dice_6,dice_66,dice_7,dice_77,dice_8,dice_88,dice_9,dice_99,dice_10,dice_1010])

###### validation ######
def eval_seg(args, eval_result_path, eval_loader):
    # logger setting
    logger_eval = Logger(osp.join(eval_result_path, 'dice_mpa_epoch.txt'), title='dice&mpa', resume=False)
    logger_eval.set_names(
          ['Dice', 'Dice_1', 'Dice_2', 'Dice_3', 'Dice_4', 'Dice_5', 'Dice_6', 'Dice_7', 'Dice_8', 'Dice_9','Dice_10',
         'mpa', 'pa_1', 'pa_2','pa_3', 'pa_4', 'pa_5', 'pa_6', 'pa_7', 'pa_8', 'pa_9','pa_10',])
    # load the model
    print('Loading eval model: {}'.format(args.name))
    net = net_builder(args.name)
    model = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load(args.seg_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded!')
    cudnn.benchmark = True
    # evaluate the model on validation set
    eval('eval', args, eval_loader, model, eval_result_path, logger_eval)

###### test ######
def test_seg(args, test_result_path, test_loader):
    # logger setting
    logger_test = Logger(osp.join(test_result_path, 'dice_mpa_epoch.txt'), title='dice&mpa', resume=False)
    logger_test.set_names(
          ['Dice', 'Dice_1', 'Dice_2', 'Dice_3', 'Dice_4', 'Dice_5', 'Dice_6', 'Dice_7', 'Dice_8', 'Dice_9','Dice_10',
         'mpa', 'pa_1', 'pa_2','pa_3', 'pa_4', 'pa_5', 'pa_6', 'pa_7', 'pa_8', 'pa_9','pa_10',])
    # load the model
    print('Loading test model ...')
    net = net_builder(args.name)
    model = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load(args.seg_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded!')
    cudnn.benchmark = True
    # test the model on testing set
    eval('test', args, test_loader, model, test_result_path, logger_test)

def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # config
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('--name', dest='name',help='change model',default=None, type=str)
    parser.add_argument('-j', '--workers', type=int, default=2)
    # train setting
    parser.add_argument('--step', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--loss',help='change model',default='wce', type=str) 
    parser.add_argument('--t', type=str, default='t1')
    parser.add_argument('--seg-path', help='pretrained model test', default=' ', type=str)
    args = parser.parse_args()
    return args

def main():
    ##### config #####
    args = parse_args()
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('torch version:',torch.__version__)
    ##### result path setting #####
    for t in range(1):
        t=t+3
        tn = args.t + '_' + str(t)
        task_name = args.data_dir.split('/')[-2] + '/' + args.data_dir.split('/')[-1]
        train_result_path = osp.join('result',task_name,'train',args.name + '_' +str(args.lr) + '_'+ tn)
        if not exists(train_result_path):
            os.makedirs(train_result_path)
        eval_result_path = osp.join('result',task_name,'eval',args.name + '_' +str(args.lr) + '_'+ tn)
        if not exists(eval_result_path):
            os.makedirs(eval_result_path)
        test_result_path = osp.join('result',task_name,'test',args.name + '_' +str(args.lr) + '_'+ tn)
        if not exists(test_result_path):
            os.makedirs(test_result_path)
        ##### load dataset #####
        info = json.load(open(osp.join(args.data_dir, 'info.json'), 'r'))
        normalize = dt.Normalize(mean=info['mean'], std=info['std'])
        t = []
        t.extend([dt.Label_Transform(),dt.ToTensor(),normalize])
        train_dataset = SegList(args.data_dir, 'train', dt.Compose(t))
        val_dataset = SegList(args.data_dir, 'eval', dt.Compose(t))
        test_dataset = SegList(args.data_dir, 'test', dt.Compose(t))
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
        eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)
        ##### train #####
        #train_seg(args,train_result_path,train_loader,eval_loader)
        ##### test #####
        model_best_path = osp.join(osp.join(train_result_path,'model'),'model_best.pth.tar')
        args.seg_path = model_best_path
        #eval_seg(args,eval_result_path,eval_loader)
        test_seg(args,test_result_path,test_loader)

if __name__ == '__main__':
    main()
