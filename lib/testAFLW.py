import cv2, os
import importlib
import sys
import logging
import torch
import torchvision.models as models
from functions import *
from networks import *
import torchvision.transforms as transforms

sys.path.insert(0, '..')


if __name__ == '__main__':  
    experiment_name = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
    data_name = 'nAFLW'
    test_labels = 'test.txt'
    test_images = 'images_test'
    config_path = '.experiments.{}.{}'.format(data_name, experiment_name)

    my_config = importlib.import_module(config_path, package='PIPNet')
    Config = getattr(my_config, 'Config')
    cfg = Config()
    cfg.experiment_name = experiment_name
    cfg.data_name = data_name

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

    if not os.path.exists(os.path.join('./logs', cfg.data_name)):
        os.mkdir(os.path.join('./logs', cfg.data_name))
    log_dir = os.path.join('./logs', cfg.data_name, cfg.experiment_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)

    save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)
    resnet18 = models.resnet18(pretrained=cfg.pretrained)
    net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride).cuda()
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_file = 'D:/project/PIPNet/snapshots/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch29.pth'
    #weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
    state_dict = torch.load(weight_file, map_location=device)
    net.load_state_dict(state_dict)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])
    norm_indices = None
    labels = get_label(cfg.data_name, test_labels)
    nmes_std = []
    nmes_merge = []
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
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb)
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        t2 = time.time()
        time_all += (t2-t1)
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()
        #standard
        nme_std = compute_nme(lms_pred, lms_gt, norm)
        nmes_std.append(nme_std)
        #merge
        nme_merge = compute_nme(lms_pred_merge, lms_gt, norm)
        nmes_merge.append(nme_merge)
    print('Total inference time:', time_all)
    print('Image num:', len(labels))
    print('Average inference time:', time_all/len(labels))

    print('nme: {}'.format(np.mean(nmes_merge)))
    logging.info('nme: {}'.format(np.mean(nmes_merge)))

    fr, auc = compute_fr_and_auc(nmes_merge)
    print('fr : {}'.format(fr))
    logging.info('fr : {}'.format(fr))
    print('auc: {}'.format(auc))
    logging.info('auc: {}'.format(auc))



