#root_directory="/home/pecs/tartanvo/data/TUM/"
root_directory="/media/scratch/TUMVI/"

# Run the python code inside Dataset folder
# cd "/home/pecs/tartanvo/Datasets"
# python3 processTUM.py 

cd "/home/pecs/SelectiveAR/tartanvo/"

#ateCSVfile="results/ate.csv"
ateCSVfile="results_given/ate_given.csv"

mkdir -p results_given

echo "dataset,ate,KITTI-R,KITTI-T" > $ateCSVfile

for folder in $root_directory/*; do
    folder_name=$root_directory$(basename $folder)
    echo $folder_name
    # If not a directory, skip
    if [ ! -d "$folder_name" ]; then
        continue
    fi

    #number=$(echo "$folder_name" | sed -n 's/.*freiburg\([0-9]\+\).*/\1/p')
    
    # To run the actual code
    # python3 vo_trajectory_from_folder.py  --model-name tartanvo_1914.pkl \
    #                                 --tum"$number" \
    #                                 --batch-size 1 --worker-num 1 \
    #                                 --test-dir  "$folder_name/rgb"\
    #                                 --pose-file "$folder_name/alter_groundtruth.txt"\
    #                                   --quiet

    # To run the code with reprojection
    # python3 ICPReprojection.py --test-dir "$folder_name/rgb" \
    #                         --tum"$number" \
    #                         --gt_pose "$folder_name/alter_groundtruth.txt" \
    #                         --model-name tartanvo_1914.pkl \
    #                         --pred_pose "results/tum"$number"_tartanvo_1914_"$(basename $folder)".txt" \
    #                         --no-decompose

    # Run the code for tum vi dataset -- without the inertial unit
    #--test-dir  "$folder_name/dso/cam0/images"\
    # Considering undistorted images
    # NOTE: To include clahe -- add --clahe
    python3 vo_trajectory_from_folder.py  --model-name tartanvo_1914.pkl \
                                    --tumvi \
                                    --batch-size 1 --worker-num 1 \
                                    --test-dir  "$folder_name/dso/cam0/given_undistorted"\
                                    --pose-file "$folder_name/alter_groundtruth.txt" \
                                    --csv-file $ateCSVfile \
                                    --quiet

done
