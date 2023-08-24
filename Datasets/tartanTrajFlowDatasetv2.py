# This traversal function takes as input the 

import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir
from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer

class TrajFolderDatasetv2(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, imgfolder , predPoseFile = None):
        
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        print(len(self.rgbfiles))
        self.rgbfiles.sort()
        self.imgfolder = imgfolder

        print('Find {} image files in {}'.format(len(self.rgbfiles), imgfolder))

        self.poses = self.poseToMotion(predPoseFile)
    
        self.N = len(self.rgbfiles) - 1

    def poseToMotion(self, posefile):
        poselist = np.loadtxt(posefile).astype(np.float32)
        assert(poselist.shape[1]==7) # position + quaternion
        poses = pos_quats2SEs(poselist)
        return poses

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        imgfile1 = self.rgbfiles[idx].strip()
        imgfile2 = self.rgbfiles[idx+1].strip()
        img1 = cv2.imread(imgfile1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(imgfile2, cv2.IMREAD_GRAYSCALE)
      
        res = {'img1': img1, 'img2': img2 }
        res['PredMat1'] = self.poses[idx]
        res['PredMat2'] = self.poses[idx+1]

        return res


