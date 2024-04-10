import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir
from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer

class TrajFolderDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, imgfolder , posefile = None, transform = None, 
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0, clahe=False):
        
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        # print(len(self.rgbfiles))
        self.rgbfiles.sort()
        self.imgfolder = imgfolder

        # print('Find {} image files in {}'.format(len(self.rgbfiles), imgfolder))

        if posefile is not None and posefile!="":
            print('Loading pose file: ', posefile)
            poselist = np.loadtxt(posefile).astype(np.float32)
            # print('Shape of pose list: ', poselist.shape)
            assert(poselist.shape[1]==7) # position + quaternion
            poses = pos_quats2SEs(poselist)
            self.matrix = pose2motion(poses)
            self.motions     = SEs2ses(self.matrix).astype(np.float32)
            # self.motions = self.motions / self.pose_std
            #assert(len(self.motions) == len(self.rgbfiles)) - 1
            # Changed to handle the cases in which the images are more and the gt data is less
            self.N = min(len(self.rgbfiles), len(self.motions))

        else:
            self.motions = None
            self.N = len(self.rgbfiles) - 1

        # self.N = len(self.lines)
        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery
        self.clahe = clahe

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        imgfile1 = self.rgbfiles[idx].strip()
        imgfile2 = self.rgbfiles[idx+1].strip()

        timestamp = imgfile2.split('/')[-1].split('.')[0]

        img1 = cv2.imread(imgfile1)
        img2 = cv2.imread(imgfile2)

        if self.clahe:
            img1_lab = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
            img2_lab = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
            #clahe = cv2.createCLAHE(4.0, (32,32))
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            # Applying to all channels
            img1_lab[...,0] = clahe.apply(img1_lab[...,0])
            img2_lab[...,0] = clahe.apply(img2_lab[...,0])

            img1 = cv2.cvtColor(img1_lab, cv2.COLOR_LAB2RGB)
            img2 = cv2.cvtColor(img2_lab, cv2.COLOR_LAB2RGB)
        res = {'img1': img1, 'img2': img2}

        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer

        if self.transform:
            res = self.transform(res)

        debugDict = {'imgpath1': imgfile1, 'imgpath2': imgfile2}

        if self.motions is None:
            return res, timestamp, debugDict
        else:
            assert idx < len(self.motions), f"Index {idx} out of range > {len(self.motions)}"
            print("Motion is", self.motions[idx])
            res['motion'] = self.motions[idx]
            return res, timestamp, debugDict


