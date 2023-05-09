# Final Year Project
A Semi-Supervised trained Cervical Landmark Detector to enhance first-level diagnosis of airway difficulty. This project is built upon the people at https://github.com/jhb86253817/PIPNet. Do check out their work on semi-supervised facial landmark detection.

## Installation
1. Install Python3 and PyTorch >= v1.1
2. Clone this repository.
```Shell
git clone https://github.com/airrehn/FYP.git
```
3. Install the dependencies in requirements.txt.
```Shell
pip install -r requirements.txt
```
## Preparing Dataset
4. Download the AFLW dataset, then put them under folder `data`. The folder structure should look like this:
````
PIPNet
-- FaceBoxesV2
-- lib
-- experiments
-- logs
-- snapshots
-- data
   |-- AFLW
       |-- flickr
       |-- AFLWinfo_release.mat
````
7. Create a new folder nAFLW (n for neck) under data.
8. Take ```EDITTED_labels.txt``` file from root and put into this folder (you can delete it from the root after). Copy all the AFLW images (from the 3 folders of flicker) and put it into one "image" folder under nAFLW. It should contain 21,123 images in total.
````
PIPNet
-- FaceBoxesV2
-- lib
-- experiments
-- logs
-- snapshots
-- data
   |-- nAFLW
       |-- images
       |-- EDITTED_labels.txt
````
9. Go to folder `lib`, do ```python preprocess256.py```.
10. Then do ```python preprocessAFLW.py```.
11. Next in the nAFLW folder, create a new folder called `images_train+aflw`, then copy and paste all images from the ```images_256``` and `images_train` folder into it.
````
PIPNet
-- FaceBoxesV2
-- lib
-- experiments
-- logs
-- snapshots
-- data
   |-- nAFLW
       |-- images
       |-- images256
       |-- EDITTED_labels.txt
       |-- images_test
       |-- images_train
       |-- images_train+aflw
       |-- meanface.txt
       |-- test.txt
       |-- train.txt
````
## Training and Testing models

12. Before training/testing/demo, check out the configuration files in ```experiments``` for example, ```experiments/nAFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py```. Then read the 4 `.sh` script files to understand the arguments taken.
14. Now you can perform ```sh run_train.sh``` for a fully supervised model or ```sh run_trainSSL.sh``` for a semi-supervised model. Remember to edit the sh file first to choose which ```config``` file you want to run. Models are saved in ```snapshots```. Logging infomation is found in ```logs```.
15. Similarly, testing a trained model can be done with ```sh run_test.sh```, do edit the sh file to select your model. And demo-ing a model can be done with ```sh run_demo.sh```, do edit the sh file to select your model and folder of unseen images.
