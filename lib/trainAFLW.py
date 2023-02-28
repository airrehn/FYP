import importlib
import os
import logging
from functions import * 
import torchvision.models as models
import torch.nn as nn
from networks import *
import torch.optim as optim
import data_utils
import torchvision.transforms as transforms
import sys
sys.path.insert(0, '..')




if __name__ == '__main__':  
    experiment_name = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
    data_name = 'nAFLW'
    config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
    
    my_config = importlib.import_module(config_path, package='PIPNet')
    Config = getattr(my_config, 'Config')
    cfg = Config()
    cfg.experiment_name = experiment_name
    cfg.data_name = data_name

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

    if not os.path.exists(os.path.join('./snapshots', cfg.data_name)):
        os.mkdir(os.path.join('./snapshots', cfg.data_name))
    save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(os.path.join('./logs', cfg.data_name)):
        os.mkdir(os.path.join('./logs', cfg.data_name))
    log_dir = os.path.join('./logs', cfg.data_name, cfg.experiment_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)

    print('###########################################')
    print('experiment_name:', cfg.experiment_name)
    print('data_name:', cfg.data_name)
    print('det_head:', cfg.det_head)
    print('net_stride:', cfg.net_stride)
    print('batch_size:', cfg.batch_size)
    print('init_lr:', cfg.init_lr)
    print('num_epochs:', cfg.num_epochs)
    print('decay_steps:', cfg.decay_steps)
    print('input_size:', cfg.input_size)
    print('backbone:', cfg.backbone)
    print('pretrained:', cfg.pretrained)
    print('criterion_cls:', cfg.criterion_cls)
    print('criterion_reg:', cfg.criterion_reg)
    print('cls_loss_weight:', cfg.cls_loss_weight)
    print('reg_loss_weight:', cfg.reg_loss_weight)
    print('num_lms:', cfg.num_lms)
    print('save_interval:', cfg.save_interval)
    print('num_nb:', cfg.num_nb)
    print('use_gpu:', cfg.use_gpu)
    print('gpu_id:', cfg.gpu_id)
    print('###########################################')
    logging.info('###########################################')
    logging.info('experiment_name: {}'.format(cfg.experiment_name))
    logging.info('data_name: {}'.format(cfg.data_name))
    logging.info('det_head: {}'.format(cfg.det_head))
    logging.info('net_stride: {}'.format(cfg.net_stride))
    logging.info('batch_size: {}'.format(cfg.batch_size))
    logging.info('init_lr: {}'.format(cfg.init_lr))
    logging.info('num_epochs: {}'.format(cfg.num_epochs))
    logging.info('decay_steps: {}'.format(cfg.decay_steps))
    logging.info('input_size: {}'.format(cfg.input_size))
    logging.info('backbone: {}'.format(cfg.backbone))
    logging.info('pretrained: {}'.format(cfg.pretrained))
    logging.info('criterion_cls: {}'.format(cfg.criterion_cls))
    logging.info('criterion_reg: {}'.format(cfg.criterion_reg))
    logging.info('cls_loss_weight: {}'.format(cfg.cls_loss_weight))
    logging.info('reg_loss_weight: {}'.format(cfg.reg_loss_weight))
    logging.info('num_lms: {}'.format(cfg.num_lms))
    logging.info('save_interval: {}'.format(cfg.save_interval))
    logging.info('num_nb: {}'.format(cfg.num_nb))
    logging.info('use_gpu: {}'.format(cfg.use_gpu))
    logging.info('gpu_id: {}'.format(cfg.gpu_id))
    logging.info('###########################################')


    meanface_indices, _, _, _ = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

    resnet18 = models.resnet18(pretrained=cfg.pretrained)
    net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride).cuda()
    print(torch.cuda.is_available())
    #device = torch.device("cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion_cls = nn.MSELoss()
    criterion_reg = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)

    points_flip = [2,1,3,4,5] #if image flipped horizontally, how does the new points look like. 1 & 2 swapped, 345 are symmetry.
    points_flip = (np.array(points_flip)-1).tolist()
    assert len(points_flip) == 5
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    labels = get_label(cfg.data_name, 'train.txt')
    train_data = data_utils.ImageFolder_pip(os.path.join('data', cfg.data_name, 'images_train'), 
                                        labels, cfg.input_size, cfg.num_lms, 
                                        cfg.net_stride, points_flip, meanface_indices,
                                        transforms.Compose([
                                        transforms.RandomGrayscale(0.2),
                                        transforms.ToTensor(),
                                        normalize]))

    #train_data,labels = train_data.cuda(), labels.cuda()

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    train_model(cfg.det_head, net, train_loader, criterion_cls, criterion_reg, cfg.cls_loss_weight, cfg.reg_loss_weight, cfg.num_nb, optimizer, cfg.num_epochs, scheduler, save_dir, cfg.save_interval, device)



