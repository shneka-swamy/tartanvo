import argparse
from pathlib import Path
from tqdm import tqdm
import glob

def argparser():
    parser = argparse.ArgumentParser(description='TUM Dataset')
    parser.add_argument('--directory', type=Path, default='/home/pecs/tartanvo/data/TUM')
    return parser.parse_args()

def chageGroundTruth(folder, groundtruth, rgb):
    with open(groundtruth, 'r') as f:
        lines = f.readlines()
    with open(rgb, 'r') as f:
        rgb_lines = f.readlines()
    
    changed_gt = []
    i = 0
    # Compare the timestamp to get the nearest groundtruth
    for rgb_line in rgb_lines:
        if rgb_line[0] == '#':
            continue
        rgb_timestamp = float(rgb_line.split(' ')[0])
        
        while i < len(lines):
            assert len(lines[i]) != 0, f"Line {i} is empty"
            if lines[i][0] == '#':
                i += 1
                continue

            #gt_timestamp = float(lines[i].split(' ')[0])
            # For tum vi dataset
            gt_timestamp = float(lines[i].split(',')[0]) 
            while gt_timestamp < rgb_timestamp and i == 0:
                gt_timestamp = float(lines[i].split(',')[0])
        
            if gt_timestamp >= rgb_timestamp:
                if lines[i-1][0] == '#':    
                    # Remove the timestamp
                    #line_list = lines[i].split(' ')[1:]
                    line_list = lines[i].split(',')[1:]
                    line_str = str.join(' ', line_list)
                    assert len(line_str) != 0, f"Line is empty"
                    changed_gt.append(line_str)
                    break

                #diff_prev = rgb_timestamp - float(lines[i-1].split(' ')[0])
                diff_prev = rgb_timestamp - float(lines[i].split(',')[0]) 

                diff_next = gt_timestamp - rgb_timestamp
                if diff_prev < diff_next:
                    line_list = lines[i-1].split(',')[1:]
                    line_str = str.join(' ', line_list)
                    assert len(line_str) != 0, f"Line is empty with {i-1}"
                    changed_gt.append(line_str)
                    i = i - 1
                else:
                    line_list = lines[i].split(',')[1:]
                    line_str = str.join(' ', line_list)
                    assert len(line_str) != 0, f"Line is empty with {i}"
                    changed_gt.append(line_str)
                break
            i += 1
    
    changed_rgb = []
    for line in rgb_lines:
        if line[0] == '#':
            continue
        changed_rgb.append(line)

    minimum_length = min(len(changed_gt), len(changed_rgb))
    changed_gt = changed_gt[:minimum_length]
    changed_rgb = changed_rgb[:minimum_length]

    assert len(changed_gt) == len(changed_rgb), "The length of the groundtruth and rgb should be the same"
    print("The length after changing")
    print(len(changed_gt), len(changed_rgb))

    with open(folder/ 'alter_rgb.txt', 'w') as f:
        for line in changed_rgb:
            f.write(line)

    with open(folder/ 'alter_groundtruth.txt', 'w') as f:
        for line in changed_gt:
            f.write(line)
    
    with open(folder/'alter_gt_with_time.txt', 'w') as f:
        for i in range(len(changed_gt)):
            f.write(changed_rgb[i].split(' ')[0] + ' ' + changed_gt[i])


def generateAlterGroundTruth(folder):
    groundtruth = folder/ 'dso' / 'gt_imu.csv'
    rgb = folder/ 'dso'/ 'cam0' / 'times.txt'

    # Change the groundtruth
    chageGroundTruth(folder, groundtruth, rgb)

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
        # Get the groundtruth and rgb file (for tum mono dataset)
        # groundtruth = folder / 'groundtruth.txt'
        # rgb = folder / 'rgb.txt'
        # Get the groundtruth and rgb file (for tum vi dataset)
        generateAlterGroundTruth(folder)

if __name__ == '__main__':
    main()