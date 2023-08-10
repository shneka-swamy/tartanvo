# Download TUM dataset
import urllib.request as urllib3
import tarfile
import requests
from multiprocessing import Pool, Manager
from functools import partial
from pathlib import Path

# Download the dataset and unzip the file
def downloadDataset(url, fileDir, print_lock):
    filename = fileDir / url.split('/')[-1]
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file:
            downloaded_size = 0
            for data in response.iter_content(chunk_size=1024):
                downloaded_size += len(data)
                file.write(data)
                percent = int(downloaded_size * 100 / total_size)
                with print_lock:
                    print(f"Downloading {filename} : {percent}% ({downloaded_size}/{total_size} bytes)", end='\r', flush=True)
        
        print("\nDownload complete!")

        # unzip a tgz file
        with tarfile.open(filename, 'r:gz') as tar_ref:
            tar_ref.extractall(fileDir)
        
        with print_lock:
            print('Finished downloading and unzipping:', url)
    except Exception as e:
        with print_lock:
            print('Error downloading:', url)
            print(e)

def main():
    fileDir = Path('../data/TUM/')
    # Download the hand    
    tum_url = [
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_360.tgz',
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_floor.tgz',
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz',
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz',
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz',
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_360_hemisphere.tgz',    
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_360_kidnap.tgz',
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz',
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_large_no_loop.tgz',
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_large_with_loop.tgz',
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz', 
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_360.tgz',
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam.tgz'
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam2.tgz',
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam3.tgz'  
    ]
  

    test = [
        'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz',
        #'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_rpy.tgz'
    ]


    # To download in parallel using multiprocessing
    
    with Manager() as manager:
        print_lock = manager.Lock()
        downloadDatasetFunc = partial(downloadDataset, fileDir=fileDir, print_lock=print_lock)

        pool = Pool(processes=4)
        pool.map(downloadDatasetFunc, test)
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()