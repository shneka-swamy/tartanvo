# Our implementation from the previously provided code

import cv2
import numpy as np
import argparse
from tqdm import tqdm
import glob
import os

def argparser():
    parser = argparse.ArgumentParser(description='Undistort images')
    parser.add_argument('--input', type=str, help='Input image folder', 
                        default='/media/scratch/TUMVI/')

    return parser.parse_args()

def undistort_image(imageD):
    # Load the image
    # Pinhole camera parameters
    intri = [190.97847715128717, 190.9733070521226, 254.93170605935475, 256.8974428996504]
    D = [0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182]
    fu, fv, pu, pv = intri

    # Create the camera matrix
    K = np.array([[fu, 0, pu],
                [0, fv, pv],
                [0, 0, 1]])
    D = np.array(D)

    # Undistort the image
    unImg = cv2.fisheye.undistortImage(imageD, K, D=D)

    return unImg


def main():
    args = argparser()
    imgfolder = args.input
    folders = glob.glob(imgfolder + '*')

    for folder in folders:
        # Skip files that are not folders
        if not os.path.isdir(folder):
            continue

        print("Processing folder: ", folder)
        path = folder + '/dso/cam0/images/'
        files = glob.glob(path + '*.png')

        for file in tqdm(files):
            image = cv2.imread(file)
            image = undistort_image(image)
            filename = os.path.basename(file)
            path = folder + '/dso/cam0/undistorted/'
            if not os.path.exists(path):
                os.makedirs(path)

            cv2.imwrite(path + filename, image)

if __name__ == '__main__':
    main()