# semi-supervised learning TRAIN (1600 labelled neck + 21k unlabelled aflw images)

#UNCOMMENT THE ONE CONFIG FILE YOU WANT TO RUN.

# nAFLW, resnet18
python lib/trainAFLW_ssl.py experiments/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py

# nAFLW, resnet50
#python lib/trainAFLW_ssl.py experiments/nAFLW/pip_32_16_60_r50_l2_l1_10_1_nb10.py

# nAFLW, resnet101
#python lib/trainAFLW_ssl.py experiments/nAFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py