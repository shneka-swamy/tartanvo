from pathlib import Path

def main():
    result_file_path = Path('/home/pecs/SelectiveAR/tartanvo/results')
    result_file = Path('dataset-corridor1_512_16.txt')

    gt_file = Path('/media/scratch/TUMVI/dataset-corridor1_512_16/alter_gt_with_time.txt')

    # Open both results and groundtruth
    with open(result_file_path / result_file, 'r') as f:
        result_lines = f.readlines()
    with open(gt_file, 'r') as f:
        gt_lines = f.readlines()
    
    changed_result = []
    for i in range(len(result_lines)):
        gt_timestamp = float(gt_lines[i].split(' ')[0])
        results_values = result_lines[i].split(' ')
        results_values.insert(0, gt_timestamp)
        results_values = [float(value) for value in results_values]
        changed_result.append(results_values)
    
    with open(result_file_path / 'altered_dataset-corridor1_512_16.txt', 'w') as f:
        for line in changed_result:
            line_str = str.join(' ', [str(value) for value in line])
            line_str += '\n'
            f.write(line_str)


if __name__ == '__main__':
    main()