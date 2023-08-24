# The results of the trajectory projected is again reprojected
# Thus, this file is done close to the test folder 
import argparse
import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R

import ICPOptimsation as icp

from Datasets.transformation import so2quat
from Datasets.tartanTrajFlowDatasetv2 import TrajFolderDatasetv2
from Datasets.utils import dataset_intrinsics
import matplotlib
import torch
import math

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--tum1', action='store_true', default=False,
                        help='tum test (default: False)')
    parser.add_argument('--tum2', action='store_true', default=False,
                    help='tum test (default: False)')
    parser.add_argument('--tum3', action='store_true', default=False,
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
    rotation1_with_trans = np.zeros((4, 4))
    rotation1_with_trans[:3, :4] = pose1.reshape(3, 4)
    rotation1_with_trans[3, 3] = 1

    rotation2_with_trans = np.zeros((4, 4))
    rotation2_with_trans[:3, :4] = pose2.reshape(3, 4)
    rotation2_with_trans[3, 3] = 1

    H = intrinsic_matrix @ rotation2_with_trans @ np.linalg.inv(rotation1_with_trans) @ np.linalg.inv(intrinsic_matrix)
    H = H[:3, :3]

    return H

def get_points_from_keypoints(keypoints):
    points = np.zeros((len(keypoints), 2))
    for i, kp in enumerate(keypoints):
        points[i, :] = kp.pt
    return points

def transform_pose(img1, img2, pose1, pose2, intrinsic_matrix):
    # Find the keypoints and descriptors with ORB
    orb = cv2.ORB_create() 
    kp1 = orb.detect(img1, None)
    kp2 = orb.detect(img2, None)

    homography_initial_guess = get_homography_from_pose(intrinsic_matrix, pose1, pose2)

    points1 = get_points_from_keypoints(kp1)
    points2 = get_points_from_keypoints(kp2)
    
    if points1.shape[0] < 100 or points2.shape[0] < 100:
        return homography_initial_guess
    else:
        min_points = min(points1.shape[0], points2.shape[0])
        points1 = points1[:min_points, :]
        points2 = points2[:min_points, :]
        T, _, _ = icp.icp(points1, points2, homography_initial_guess)
        return T

def detect_quartenion(rvec, tvec, pose1, pose2):
    lowest_change = math.inf
    required_rotation = [0, 0, 0, 0]
    required_translation = [0, 0, 0]
    # Choose the one that is closest to the one that is predicted
    for rot, trans in zip(rvec, tvec):
        rotation2_with_trans = np.zeros((4, 4))
        rotation2_with_trans[:3, :4] = pose2.reshape(3, 4)
        rotation2_with_trans[3, 3] = 1

        relative_rotation = np.zeros((4, 4))
        relative_rotation[:3, :3] = rot
        relative_rotation[:3, 3] = trans.reshape(3)
        relative_rotation[3, 3] = 1

        rotation2 = relative_rotation @ rotation2_with_trans
        rotation2_pred = pose1.reshape(3, 4)
        rotation2_pred = np.vstack((rotation2_pred, np.array([0, 0, 0, 1])))

        change = np.linalg.norm(rotation2 - rotation2_pred, ord=2) 
        if change < lowest_change:
            lowest_change = change
            required_rotation = R.from_matrix(rotation2[:3, :3]).as_quat()
            required_translation = rotation2[:3, 3]

    return required_rotation, required_translation  
    
def data_loader(arg):
    matplotlib.use('Agg')
    datastr = 'tartanair'
    if arg.tum1:
        datastr = 'tum1'
    elif arg.tum2:
        datastr = 'tum2'
    elif arg.tum3:
        datastr = 'tum3'

    focalx, focaly, centerx, centery = dataset_intrinsics(datastr)

    intrinsic_matrix = np.array([[focalx, 0.0, centerx, 0.0],
                                 [0.0, focaly, centery, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])

    print("Got the intrinsic matrix")

    reproject_dataset = TrajFolderDatasetv2(arg.test_dir, arg.pred_pose)
    print("Got the reprojection dataset")
    reproject_loader = torch.utils.data.DataLoader(reproject_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    reprojectIter = iter(reproject_loader) 
    while True:
        try:
            res = next(reprojectIter)
            # Convert torch tensor to numpy array
            img1 = res['img1'].numpy()
            img1 = np.squeeze(img1, axis=0)
            img2 = res['img2'].numpy()
            img2 = np.squeeze(img2, axis=0)
            transformed_value = transform_pose(img2, img1, res['PredMat2'], res['PredMat1'], intrinsic_matrix)
            _, rvec, tvec, _ = cv2.decomposeHomographyMat(transformed_value, intrinsic_matrix[:3, :3])
            
            required_rotation, required_translation = detect_quartenion(rvec, tvec, res['PredMat2'], res['PredMat1'])
            print(f"Required rotation: {required_rotation}, Required translation: {required_translation}")
        except StopIteration:
            break

def main(arg):
    data_loader(arg)

if __name__ == '__main__':
    arg = get_args()
    main(arg)