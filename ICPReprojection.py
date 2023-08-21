# The results of the trajectory projected is again reprojected
# Thus, this file is done close to the test folder 


import argparse
from tqdm import tqdm
import numpy as np
import cv2

from arguments import commandParser

from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import IterativeClosestPoint as icp

from Datasets.tartanTrajFlowDatasetv2 import TrajFolderDatasetv2
from Datasets.utils import dataset_intrinsics
import matplotlib
import torch

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--tum', action='store_true', default=False,
                        help='tum test (default: False)')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--gt_pose', default='',
                        help='path to the ground truth pose file')
    parser.add_argument('--pred_pose', default='',
                        help='path to the predicted pose file')
    args = parser.parse_args()

    return args

def get_homography_from_pose(intrinsic_matrix, pose1, pose2):
    translation1 = pose1[0][:3]
    translation2 = pose2[0][:3]

    rotation1 = R.from_quat(pose1[0][3:]).as_matrix()
    rotation2 = R.from_quat(pose2[0][3:]).as_matrix()

    rotation1_with_trans = np.zeros((4, 4))
    rotation1_with_trans[:3, :3] = rotation1
    rotation1_with_trans[:3, 3] = translation1
    rotation1_with_trans[3, 3] = 1.0

    rotation2_with_trans = np.zeros((4, 4))
    rotation2_with_trans[:3, :3] = rotation2
    rotation2_with_trans[:3, 3] = translation2
    rotation2_with_trans[3, 3] = 1.0
   
    H = intrinsic_matrix @ rotation2_with_trans @ np.linalg.inv(rotation1_with_trans) @ np.linalg.inv(intrinsic_matrix)
    H = H[:3, :3]

    print("Homography matrix from pose:", H)
    return H

def get_points_from_keypoints(keypoints):
    points = np.zeros((len(keypoints), 2))
    for i, kp in enumerate(keypoints):
        points[i, :] = kp.pt
    return points

def transform_pose(img1, img2, pose1, pose2):
    # Find the keypoints and descriptors with ORB
    orb = cv2.ORB_create() 
    kp1 = orb.detect(img1, None)
    kp2 = orb.detect(img2, None)

    homography_initial_guess = get_homography_from_pose(pose1, pose2)

    points1 = get_points_from_keypoints(kp1)
    points2 = get_points_from_keypoints(kp2)

    #print(f"Points1: {points1.shape}, Points2: {points2.shape}")
    T, _, _ = icp.icp(points1, points2, homography_initial_guess)
    return T


def data_loader(arg):
    matplotlib.use('Agg')
    datastr = 'tartanair'
    if arg.tum:
        datastr = 'tum'

    focalx, focaly, centerx, centery = dataset_intrinsics(datastr)

    intrinsic_matrix = np.array([[focalx, 0.0, centerx, 0.0],
                                 [0.0, focaly, centery, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])

    reproject_dataset = TrajFolderDatasetv2(arg.test_dir, arg.pred_pose)
    reproject_loader = torch.utils.data.DataLoader(reproject_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    reprojectIter = iter(reproject_loader) 
    while True:
        try:
            res = next(reprojectIter)
            transform_pose(res['img1'], res['img2'], res['pose1'], res['pose2'])    
            break
        except StopIteration:
            break

def main(arg):
    data_loader(arg)

if __name__ == '__main__':
    arg = commandParser()
    main(arg)