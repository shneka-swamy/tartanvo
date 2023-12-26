# The results of the trajectory projected is again reprojected
# Thus, this file is done close to the test folder 
import argparse
import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R

import ICPOptimsation as icp

from evaluator.tartanair_evaluator import TartanAirEvaluator
from Datasets.transformation import so2quat, pos_quat2SE
from Datasets.tartanTrajFlowDatasetv2 import TrajFolderDatasetv2
from Datasets.utils import dataset_intrinsics, plot_traj
import matplotlib
import torch
import math

from tqdm import tqdm
def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--tum1', action='store_true', default=False,
                        help='tum test (default: False)')
    parser.add_argument('--tum2', action='store_true', default=False,
                    help='tum test (default: False)')
    parser.add_argument('--tum3', action='store_true', default=False,
                    help='tum test (default: False)')
    parser.add_argument('--tumvi', action='store_true', default=False,
                    help='tum test (default: False)')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--gt_pose', default='',
                        help='path to the ground truth pose file')
    parser.add_argument('--pred_pose', default='',
                        help='path to the predicted pose file')
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--no-decompose', action='store_true', default=False,
                        help='do not decompose the homography matrix (default: False)')
    parser.add_argument('--from_gt', action='store_true', default=False,
                        help='use the ground truth pose as the initial value (default: False)')
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
    
    if points1.shape[0] < 200 or points2.shape[0] < 200:
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

        # Considering only translation
        change = np.linalg.norm(rotation2[:3, 3] - rotation2_pred[:3, 3], ord=2) 
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
    elif arg.tumvi:
        datastr = 'tumvi'

    poselist = []
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
    flag = True
    if arg.from_gt:
        with open(arg.gt_pose, 'r') as f:
            pose = f.readline().split()
            pose = np.array([float(p) for p in pose])
            poselist.append(pose)


    with tqdm(total=len(reproject_loader)) as pbar: 
        while True:
            try:
                res = next(reprojectIter)

                if arg.from_gt and flag:
                    res['PredMat1'] = torch.tensor(pos_quat2SE(poselist[0]))
                    flag = False

                # Convert torch tensor to numpy array
                if flag:
                    pose_matrix = res['PredMat1'].numpy().reshape(3, 4)
                    rotation_quat = R.from_matrix(pose_matrix[:3, :3]).as_quat()
                    translation = pose_matrix[:3, 3]
                    pose = np.concatenate((translation, rotation_quat))
                    poselist.append(pose)
                    flag = False
                

                img1 = res['img1'].numpy()
                img1 = np.squeeze(img1, axis=0)
                img2 = res['img2'].numpy()
                img2 = np.squeeze(img2, axis=0)
                transformed_value = transform_pose(img2, img1, res['PredMat2'], res['PredMat1'], intrinsic_matrix)
                
                if arg.no_decompose:
                    rotation1 = res['PredMat1'].numpy().reshape(3, 4)
                    rotation2 = np.linalg.inv(intrinsic_matrix[:3, :3]) @ transformed_value @ intrinsic_matrix[:3, :3] @ rotation1
                    required_rotation = R.from_matrix(rotation2[:3, :3]).as_quat()
                    required_translation = rotation2[:3, 3]
                    pose = np.concatenate((required_translation, required_rotation))
                    poselist.append(pose)

                else:
                    _, rvec, tvec, _ = cv2.decomposeHomographyMat(transformed_value, intrinsic_matrix[:3, :3])
                    
                    required_rotation, required_translation = detect_quartenion(rvec, tvec, res['PredMat2'], res['PredMat1'])
                    pose = np.concatenate((required_translation, required_rotation))
                    poselist.append(pose)
            
            except StopIteration:
                break
            pbar.update(1)
        
    print("Shape of the pose list: ", np.array(poselist).shape)

    testname = datastr + '_' + arg.model_name.split('.')[0] + '_' + arg.test_dir.split('/')[-2] + '_reproj'
    # calculate ATE, RPE, KITTI-RPE
    if arg.gt_pose.endswith('.txt'):
        evaluator = TartanAirEvaluator()
        print("calling evaluate trajectory")
        results = evaluator.evaluate_one_trajectory(arg.gt_pose, np.array(poselist), scale=True, kittitype=(datastr=='kitti'))
        if datastr=='euroc':
            print("==> ATE: %.4f" %(results['ate_score']))
        else:
            print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

        # save results and visualization
        plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/'+testname+'.png', title='ATE %.4f' %(results['ate_score']))
        np.savetxt('results/'+testname+'.txt',results['est_aligned'])
    else:
        np.savetxt('results/'+testname+'.txt',np.array(poselist))



def main(arg):
    data_loader(arg)

if __name__ == '__main__':
    arg = get_args()
    main(arg)