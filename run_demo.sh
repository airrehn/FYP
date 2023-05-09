# FOR EXAMPLE,
# photos are in PIPNET/images (Arguement 3), model (pth file) is in snapshots/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/SSL_nAFLW_epoch59.pth (Argument 4)




# nAFLW, resnet18
python lib/demoAFLW.py experiments/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py images snapshots/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/SSL_nAFLW_epoch59.pth

# nAFLW, resnet50
#python lib/demoAFLW.py experiments/nAFLW/pip_32_16_60_r50_l2_l1_10_1_nb10.py images snapshots/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/SSL_nAFLW_epoch59.pth

# nAFLW, resnet101
#python lib/demoAFLW.py experiments/nAFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py images snapshots/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/SSL_nAFLW_epoch59.pth