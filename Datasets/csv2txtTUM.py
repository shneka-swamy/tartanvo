import argparse
from pathlib import Path
from tqdm import tqdm
import glob

def argparser():
    parser = argparse.ArgumentParser(description='TUM Dataset')
    parser.add_argument('--directory', type=Path, default='/home/pecs/tartanvo/data/TUM')
    return parser.parse_args()

def chageGroundTruth(folder, groundtruth):
    with open(groundtruth, 'r') as f:
        lines = f.readlines()
    
    changed_gt = []
    for line in lines:
        elements = line.split(',')
        # For tum vi dataset-- convert to space sperated string
        changed_gt.append(str.join(' ', elements))  
    

    with open(folder/ 'spaceGroundTruth.txt', 'w') as f:
        for line in changed_gt:
            f.write(line)

def generateAlterGroundTruth(folder):
    groundtruth = folder/ 'dso' / 'gt_imu.csv'

    # Change the groundtruth
    chageGroundTruth(folder, groundtruth)

def main():
    args = argparser()
    directory = args.directory

    if directory.name != 'TUMVI' and directory.is_dir():
        print(f"Processing single folder {directory}")
        generateAlterGroundTruth(directory)
        return

    # Go through all the folders in the directory
    for folder in tqdm(directory.glob('*')):
        # If the folder is not a directory, skip
        if not folder.is_dir():
            continue
        print("Processing: ", folder)
        generateAlterGroundTruth(folder)

if __name__ == '__main__':
    main()