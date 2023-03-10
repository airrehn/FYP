
class Config():
    def __init__(self):
        self.det_head = 'pip'
        self.net_stride = 32
        self.batch_size = 16
        self.init_lr = 0.0001
        self.num_epochs = 15
        self.decay_steps = [30, 50]
        self.input_size = 256
        self.backbone = 'resnet18' #the 18, can change to 50 or 101 too.
        self.pretrained = True
        self.criterion_cls = 'l2'
        self.criterion_reg = 'l1'
        self.cls_loss_weight = 10
        self.reg_loss_weight = 1
        self.num_lms = 5 
        self.save_interval = self.num_epochs
        self.num_nb = 3 #change?
        self.use_gpu = True
        self.gpu_id = 0 #change? was 3
        self.curriculum = True
