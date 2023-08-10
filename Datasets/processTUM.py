import argparse

def argparser():
    parser = argparse.ArgumentParser(description='TUM Dataset')
    parser.add_argument('--groundtruth', type=str, default='/media/extra/tartanvo/data/TUM/groundtruth.txt',)
    parser.add_argument('--rgb', type=str, default='/media/extra/tartanvo/data/TUM/rgb.txt')
    return parser.parse_args()

def chageGroundTruth(groundtruth, rgb):
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
            if lines[i][0] == '#':
                i += 1
                continue

            gt_timestamp = float(lines[i].split(' ')[0])
            if gt_timestamp >= rgb_timestamp:
                if lines[i-1][0] == '#':    
                    # Remove the timestamp
                    line_list = lines[i].split(' ')[1:]
                    line_str = str.join(' ', line_list)
                    changed_gt.append(line_str)
                    break

                diff_prev = rgb_timestamp - float(lines[i-1].split(' ')[0])
                diff_next = gt_timestamp - rgb_timestamp
                if diff_prev < diff_next:
                    line_list = lines[i-1].split(' ')[1:]
                    line_str = str.join(' ', line_list)
                    changed_gt.append(line_str)
                    i = i - 1
                else:
                    line_list = lines[i].split(' ')[1:]
                    line_str = str.join(' ', line_list)
                    changed_gt.append(line_str)
                break
            i += 1
    
    changed_rgb = []
    for line in rgb_lines:
        if line[0] == '#':
            continue
        changed_rgb.append(line)
    
    assert len(changed_gt) == len(changed_rgb), "The number of groundtruth and rgb is not equal"

    with open('alter_rgb.txt', 'w') as f:
        for line in changed_rgb:
            f.write(line)

    with open('alter_groundtruth.txt', 'w') as f:
        for line in changed_gt:
            f.write(line)


def main():
    args = argparser()
    chageGroundTruth(args.groundtruth, args.rgb)

if __name__ == '__main__':
    main()