import argparse
from pathlib import Path
import math
import cv2
from tqdm import tqdm
import glob

def argparser():
    parser = parser = argparse.ArgumentParser(description='TUM Dataset')
    parser.add_argument('--tum_path', type=Path, default='/home/pecs/tartanvo/data/TUM')
    return parser.parse_args()

def converToGray(rgb):
    for folder in tqdm(rgb.glob('*')):
        rgb = folder / 'rgb'
        # Make a directory for gray images
        gray_dir = folder / 'gray'
        gray_dir.mkdir(exist_ok=True)
        # Read rgb images
        rgb_images = sorted(rgb.glob('*.png'))
        for rgb_image in tqdm(rgb_images):
            image = cv2.imread(str(rgb_image))
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(gray_dir / rgb_image.name), gray_image)


def getTime(rgb):
    for folder in tqdm(rgb.glob('*')):
        rgb_text = folder / 'rgb.txt'
        time = []
        with open(rgb_text, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == '#':
                    continue
                
                time.append(line.split(' ')[0])
        
        with open(folder / 'timestamp.txt', 'w') as f:
            for line in time:
                f.write(line +'\n')

def main():
    args = argparser()
    getTime(args.tum_path)
    converToGray(args.tum_path)

if __name__ == '__main__':
    main()