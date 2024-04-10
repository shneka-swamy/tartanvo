from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from Datasets.transformation import ses2poses_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator
from TartanVO import TartanVO

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import cv2
from os import mkdir
from os.path import isdir

from pathlib import Path
import matplotlib
from tqdm import tqdm

import csv

def get_args():
    parser = ArgumentParser(description='HRL', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--sense', action='store_true', default=False,
                        help='sense time dataset (default: False)')
    parser.add_argument('--tum1', action='store_true', default=False,
                        help='freiburg1 tum test (default: False)')
    parser.add_argument('--tum2', action='store_true', default=False,
                    help='freiburg2 tum test (default: False)')
    parser.add_argument('--tum3', action='store_true', default=False,
                    help='freiburg3 tum test (default: False)')
    parser.add_argument('--tumvi', action='store_true', default=False,
                    help='vi tum test (default: False)')
    parser.add_argument('--android', action='store_true', default=False,
                        help='android test (default: False)')
    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')
    parser.add_argument('--quiet', action='store_true', default=False, help='do not print anything (default: False)')
    parser.add_argument('--csv-file', default='./results_tartan', type=Path,
                        help='csv file to save the results (default: "")')
    parser.add_argument('--poses-file', default='poses.txt', type=Path,
                        help='file to save the estimated poses (default: "poses.txt")')
    parser.add_argument('--relative-file', default='relative.txt', type=Path,
                        help='file to save the relative poses (default: "relative.txt")')
    parser.add_argument('--clahe', action='store_true', default=False,
                        help='use clahe (default: False)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()   
    matplotlib.use('Agg')
    testvo = TartanVO(args.model_name)

    # load trajectory data from a folder
    datastr = 'tartanair'
    if args.kitti:
        datastr = 'kitti'
    elif args.euroc:
        datastr = 'euroc'
    elif args.android:
        datastr = 'android'
    elif args.tum1:
        datastr = 'tum1'
    elif args.tum2:
        datastr = 'tum2'
    elif args.tum3:
        datastr = 'tum3'
    elif args.tumvi:
        datastr = 'tumvi'
    else:
        datastr = 'tartanair'

    print("Clahe is set to: ", args.clahe)

    focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 
    if args.kitti_intrinsics_file.endswith('.txt') and datastr=='kitti':
        focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)

    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

    if datastr == 'android':
        videoname = "PXL_20230721_051431440.mp4"
        videopath = Path("./data/ANDROID") / videoname
        # read video and extract frames to Path("./data/ANDROID") / "frames" with file names 000000.png, 000001.png, ...
        # also make sure len of frames less than or equal to 999999
        extractFramePath = Path("./data/ANDROID") / "frames"
        if not (extractFramePath / "000000.png").is_file():
            cap = cv2.VideoCapture(str(videopath))
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(str(extractFramePath / (str(count).zfill(6) + ".png")), frame)
                count += 1
            cap.release()

    testDataset = TrajFolderDataset(args.test_dir,  posefile = args.pose_file, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery, clahe=args.clahe)
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
    testDataiter = iter(testDataloader)

    testDatasetName = args.test_dir.split('/')[-4]

    timelist = []
    motionlist = []
    # testname = datastr + '_' + args.model_name.split('.')[0] + '_' + args.test_dir.split('/')[-2]
    # print("TestName: ", testname)
    if args.save_flow:
        flowdir = 'results/'+testDatasetName+'_flow'
        if not isdir(flowdir):
            mkdir(flowdir)
        flowcount = 0
    with tqdm(total=len(testDataloader), desc=f"{testDatasetName}") as pbar:
        while True:
            try:
                sample, timestamp, debugDict = next(testDataiter)
            except StopIteration:
                break
    
            # cv2.imshow('img1 python', sample['img1'].numpy().squeeze().transpose(1,2,0))
            # cv2.imshow('img2 python', sample['img2'].numpy().squeeze().transpose(1,2,0))
            motions, flow = testvo.test_batch(sample, args.quiet)
            
            print("Motions are", type(motions), motions.shape, motions.dtype)
            print(motions) #, debugDict['imgpath1'], debugDict['imgpath2'])
            # cv2.waitKey(0)

            # if pbar.n > 10:
            #     exit(1)

            timelist.extend(timestamp)
            motionlist.extend(motions)

            if args.save_flow:
                for k in range(flow.shape[0]):
                    flowk = flow[k].transpose(1,2,0)
                    np.save(flowdir+'/'+str(flowcount).zfill(6)+'.npy',flowk)
                    flow_vis = visflow(flowk)
                    cv2.imwrite(flowdir+'/'+str(flowcount).zfill(6)+'.png',flow_vis)
                    flowcount += 1
            pbar.update(1)

    assert len(timelist) == len(motionlist), f"timelist: {len(timelist)}, motionlist: {len(motionlist)}"
    with open(args.relative_file, 'w') as f:
        for i in range(len(motionlist)):
            f.write(str(timelist[i])+' '+' '.join([str(x) for x in motionlist[i]])+'\n')

    poselist = ses2poses_quat(np.array(motionlist))

    print(poselist[:11])

    # calculate ATE, RPE, KITTI-RPE
    if args.pose_file.endswith('.txt'):
        evaluator = TartanAirEvaluator()
        if not args.quiet:
            print("calling evaluate trajectory")
        results = evaluator.evaluate_one_trajectory(args.pose_file, poselist, scale=True, kittitype=(datastr=='kitti'))
        if datastr=='euroc':
            csvRow = [testDatasetName, results['ate_score']]
            print("==> ATE: %.4f" %(results['ate_score']))
            print("==> RPE: %.4f" %(results['scale']))
        else:
            csvRow = [testDatasetName, results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]]
            print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

        if args.csv_file:
            with open(args.csv_file, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csvRow)

        if args.poses_file:
            timelist = [0] + timelist
            assert len(timelist) == len(poselist), f"timelist: {len(timelist)}, poselist: {len(poselist)}"
            with open(args.poses_file, 'w') as f:
                for i in range(len(poselist)):
                    f.write(str(timelist[i])+' '+' '.join([str(x) for x in poselist[i]])+'\n')

        # save results and visualization
        if args.clahe:
            plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results_clahe/'+testDatasetName+'.png', title='ATE %.4f' %(results['ate_score']))
            np.savetxt('results_clahe/'+testDatasetName+'.txt',results['est_aligned'])
        else:       
            plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results_given/'+testDatasetName+'.png', title='ATE %.4f' %(results['ate_score']))
            np.savetxt('results_given/'+testDatasetName+'.txt',results['est_aligned'])
    else:
        if args.clahe:
            np.savetxt('results_clahe/'+testDatasetName+'.txt',poselist)
        else:
            np.savetxt('results_given/'+testDatasetName+'.txt',poselist)
