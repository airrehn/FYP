# FYP
A Semi-Supervised trained Cervical Landmark Detector to enhance first-level diagnosis of airway difficulty. This project is built upon the people at https://github.com/jhb86253817/PIPNet. Do check out their work on semi-supervised facial landmark detection.

## Installation
1. Install Python3 and PyTorch >= v1.1
2. Clone this repository.
```Shell
git clone https://github.com/jhb86253817/PIPNet.git
```
3. Install the dependencies in requirements.txt.
```Shell
pip install -r requirements.txt
```
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
5. Go to folder `lib`, preprocess AFLW dataset by running ```python preprocess.py AFLW```
6. 
7. 
