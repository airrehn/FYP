import sys
import importlib
from functions import *
from networks import *
import torchvision.models as models
import torchvision.transforms as transforms

sys.path.insert(0, '..')


if __name__ == '__main__':  
    experiment_name = 'pip_32_16_60_r18_l2_l1_10_1_nb10'
    data_name = 'nAFLW'
    config_path = '.experiments.{}.{}'.format(data_name, experiment_name)

    image_folder = sys.argv[1]

    my_config = importlib.import_module(config_path, package='PIPNet')
    Config = getattr(my_config, 'Config')
    cfg = Config()
    cfg.experiment_name = experiment_name
    cfg.data_name = data_name

    save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)

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

    def demo_image(image_folder, net, preprocess, input_size, net_stride, num_nb, use_gpu, device):
        net.eval()
        for filename in os.listdir(image_folder):
            #print(filename)
            image = cv2.imread(os.path.join(image_folder,filename))
            image_height, image_width, _ = image.shape

            det_xmin = 0
            det_ymin = 0
            det_xmax = image_width-1
            det_ymax = image_height-1
            det_width = det_xmax - det_xmin + 1
            det_height = det_ymax - det_ymin + 1

            #det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
            det_crop = cv2.resize(image, (input_size, input_size))
            inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)

            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb)
            
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
            lms_pred = lms_pred.cpu().numpy()

            tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
            lms_pred_merge = lms_pred_merge.cpu().numpy()

            for i in range(cfg.num_lms):
                #unmerge and merge variation basically has smth to do with multiple faces in the image, linked to bounding box? merged seems better with a bounding box.
                x_pred = lms_pred_merge[i*2] * det_width 
                y_pred = lms_pred_merge[i*2+1] * det_height
                #x_pred = lms_pred[i*2] * det_width
                #y_pred = lms_pred[i*2+1] * det_height
                cv2.circle(image, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 2, (0, 0, 255), 2)
            cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
            cv2.imshow(filename, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    demo_image(image_folder, net, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb, cfg.use_gpu, device)