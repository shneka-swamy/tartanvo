# This is the file that is recommended by the TUMVI dataset.
# To undistort the fisheye view images

import cv2
import yaml
from matplotlib.pyplot import imshow
import numpy as np
import argparse
import glob
from tqdm import tqdm
import os

def argparser():
    parser = argparse.ArgumentParser(description='Undistort images')
    parser.add_argument('--input', type=str, help='Input image folder', 
                        default='/media/scratch/TUMVI/')

    return parser.parse_args()

def undistort(img):
    skip_lines = 0
    with open('./pinhole-equi-512/camchain-imucam-imucalib.yaml') as infile:
        for i in range(skip_lines):
            _ = infile.readline()
        data = yaml.safe_load(infile)
        
        
    # You should replace these 3 lines with the output in calibration step
    DIM=(512, 512)
    #K=np.array(YYY)
    #D=np.array(ZZZ)

    [fu, fv, pu, pv] = data['cam0']['intrinsics']
    #https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
    K = np.asarray([[fu, 0, pu], [0, fv, pv], [0, 0, 1]]) # K(3,3)
    D = np.asarray(data['cam0']['distortion_coeffs'])#D(4,1)
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted_img  
    
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
            image = undistort(image)
            filename = os.path.basename(file)
            path = folder + '/dso/cam0/given_undistorted/'
            if not os.path.exists(path):
                os.makedirs(path)

            cv2.imwrite(path + filename, image)


if __name__ == '__main__':
    main()

    
    
    img = cv2.imread("/media/scratch/TUMVI/dataset-corridor1_512_16/dso/cam0/images/1520531978006983689.png")
    undistort(img)    