# supervised learning TEST

# FOR EXAMPLE,
# model (pth file) is in snapshots/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/SSL_nAFLW_epoch59.pth (Argument 3)

#UNCOMMENT THE ONE CONFIG FILE YOU WANT TO RUN.

# nAFLW, resnet18
python lib/testAFLW.py experiments/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py snapshots/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/SSL_nAFLW_epoch59.pth 

# nAFLW, resnet50
#python lib/testAFLW.py experiments/nAFLW/pip_32_16_60_r50_l2_l1_10_1_nb10.py snapshots/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/SSL_nAFLW_epoch59.pth 

# nAFLW, resnet101
#python lib/testAFLW.py experiments/nAFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py snapshots/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/SSL_nAFLW_epoch59.pth 
