import cv2, os
import sys
sys.path.insert(0, '..')
import numpy as np
from PIL import Image
import logging
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


from networks_gssl import *
import data_utils_gssl
from functions_gssl import * 


if __name__ == '__main__': 

    if not len(sys.argv) == 2:
        print('Format:')
        print('python lib/trainAFLW_ssl.py config_file')
        exit(0)

    experiment_name = sys.argv[1].split('/')[-1][:-3]
    data_name = sys.argv[1].split('/')[-2]

    test_labels = 'test.txt'
    test_images = 'images_test'
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


    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

    resnet18 = models.resnet18(pretrained=cfg.pretrained)
    net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride).cuda()
    
    print(torch.cuda.is_available())
    #device = torch.device("cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    criterion_cls = nn.MSELoss()
    criterion_reg = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)

    points_flip = [2,1,3,4,5] #if image flipped horizontally, how the new points will look like. 1 & 2 swapped, 345 are symmetry.
    points_flip = (np.array(points_flip)-1).tolist()
    assert len(points_flip) == 5
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    labels = get_label(cfg.data_name, 'train.txt', 'std')
    train_data = data_utils_gssl.ImageFolder_pip(os.path.join('data', cfg.data_name, 'images_train+aflw'), 
                                        labels, cfg.input_size, cfg.num_lms, 
                                        cfg.net_stride, points_flip, meanface_indices,
                                        transforms.Compose([
                                        transforms.RandomGrayscale(0.2),
                                        transforms.ToTensor(),
                                        normalize]))


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    #train_model_supervised(cfg.det_head, net, train_loader, criterion_cls, criterion_reg, cfg.cls_loss_weight, cfg.reg_loss_weight, cfg.num_nb, optimizer, cfg.num_epochs, scheduler, save_dir, cfg.save_interval, device, 'pure_supervised_' + cfg.backbone)

    # IF YOU DONT WANT TO RETRAIN THE SUPERVISED PORTION EVERYTIME, you can just load in the trained model:
    # UNCOMMENT the 3 lines below, COMMENT the train_model_supervised() line above.

    weight_file = 'D:/project/PIPNet/snapshots/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/pure_supervised_resnet18_epoch19_bs96.pth' #load in the model you want
    state_dict = torch.load(weight_file, map_location=device)
    net.load_state_dict(state_dict)

    if cfg.curriculum:
    # self-training with curriculum
        task_type_list = ['cls3', 'cls2', 'std', 'std', 'std']
        STC = 'withSTC'
    else:
    # standard self-training 
        task_type_list = ['std']*3
        STC = 'noSTC'

    ##################################################################################################################################
    # initial test
    preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])
    norm_indices = None
    labels = get_label(cfg.data_name, test_labels)
    nmes = []
    norm = 1 #1 for AFLW, other datasets have other norm calculations, refer to test.py for more info.
    time_all = 0
    for label in labels:
        image_name = label[0]
        lms_gt = label[1]
        image_path = os.path.join('data', cfg.data_name, test_images, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (cfg.input_size, cfg.input_size))
        inputs = Image.fromarray(image[:,:,::-1].astype('uint8'), 'RGB')
        inputs = preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(device)
        t1 = time.time()
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb) #gets predicted landmarks
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten().cpu().numpy()
        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten().cpu().numpy()

        #############################
        nme = compute_nme(lms_pred_merge, lms_gt, norm)
        nmes.append(nme)
    print('{} nme: {}'.format('AFLW', np.mean(nmes)))
    logging.info('{} nme: {}'.format('AFLW', np.mean(nmes)))
    initialNME = np.mean(nmes)

    nmeArr = []
    for ti, task_type in enumerate(task_type_list):
        # psuedo labelling
        print('###################################################')
        print('Iter:', ti, 'task_type:', task_type)
        est_data = 'nAFLW'

        img_folder = 'images'
        est_preds = []
    
        

        image_folder = os.path.join('data', est_data, img_folder)
        for idx, filename in enumerate(os.listdir(image_folder)):
            image_name = filename
            print('%.2f %%' % (idx/len(os.listdir(image_folder)) * 100)) #loading everything can take some time, just a simple progress tracker.
            image_path = os.path.join('data', est_data, img_folder, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (cfg.input_size, cfg.input_size))
            inputs = Image.fromarray(image[:,:,::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb)
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten().cpu().numpy()
            tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten().cpu().numpy()
            est_preds.append([image_name, task_type, lms_pred_merge])
        ####################################################################################################################################
        # GSSL
        resnet18 = models.resnet18(pretrained=cfg.pretrained)
        net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride).cuda()
        optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)
        labels = get_label(cfg.data_name, 'train.txt', 'std')
        labels += est_preds #the labels also contain from the aflw dataset now, need fetch those pictures too for the retraining iterations.
        train_data = data_utils_gssl.ImageFolder_pip(os.path.join('data', cfg.data_name, 'images_train+aflw'), 
                                                labels, cfg.input_size, cfg.num_lms, 
                                                cfg.net_stride, points_flip, meanface_indices,
                                                transforms.Compose([
                                                transforms.RandomGrayscale(0.2),
                                                transforms.ToTensor(),
                                                normalize]))
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        
        train_model(cfg.det_head, net, train_loader, criterion_cls, criterion_reg, cfg.cls_loss_weight, cfg.reg_loss_weight, cfg.num_nb, optimizer, cfg.num_epochs, scheduler, save_dir, cfg.save_interval, device, 'SSL_%s' %data_name, ti)
        ####################################################################################################################################
        # test again
        preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])
        norm_indices = None
        labels = get_label(cfg.data_name, test_labels)
        nmes = []
        norm = 1 #Typically would be interoccular distance, but PIPNet documentation says use image size for AFLW. Since landmarks are scaled between 0 to 1. Then image size is just 1.
        time_all = 0
        for label in labels:
            image_name = label[0]
            lms_gt = label[1]
            image_path = os.path.join('data', cfg.data_name, test_images, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (cfg.input_size, cfg.input_size))
            inputs = Image.fromarray(image[:,:,::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
            t1 = time.time()
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb)
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten().cpu().numpy()
            tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten().cpu().numpy()

            #############################
            nme = compute_nme(lms_pred_merge, lms_gt, norm)
            nmes.append(nme)
        print('{} nme: {}'.format('AFLW', np.mean(nmes)))
        logging.info('{} nme: {}'.format('AFLW', np.mean(nmes))) 
        nmeArr.append(np.mean(nmes) * 100) #convert to percent

    plt.cla()
    nmeArr.insert(0,initialNME * 100)
    plt.plot(range(len(task_type_list)+1), nmeArr)
    plt.xlabel("Re-train")
    plt.xticks(np.arange(0,len(nmeArr),1))
    # plt.xlim(1,None)
    plt.ylabel("NME (%)")
    plt.title("Normalized Mean Error vs Re-training")

    if not os.path.exists('./graphs'):
        os.makedirs('./graphs')
    graphfilename = os.path.join('./graphs',"%s_nme_%d_%s.png" %(cfg.backbone,cfg.num_epochs,STC))
    plt.savefig(graphfilename)



