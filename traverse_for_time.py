import re

def traverse_for_time(path):
    total_time = 0
    number_of_images = 0
    with open (path, 'r') as f:
        line = f.readline()
        while line:
            if re.findall('inference using', line):
                time = float(line.strip().split(' ')[-1][:-2])
                total_time += time
                number_of_images += 1
            line = f.readline()
    print('total time: ', total_time)
    print('number of images: ', number_of_images)
    print('average time: ', total_time/number_of_images)


def main():
    path = './out.txt'
    traverse_for_time(path)

if __name__ == '__main__':
    main()