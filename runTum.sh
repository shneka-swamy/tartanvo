root_directory="/home/pecs/tartanvo/data/TUM/"

# Run the python code inside Dataset folder
cd "/home/pecs/tartanvo/Datasets"
python3 processTUM.py 

cd "/home/pecs/tartanvo/"
for folder in $root_directory/*; do
    folder_name=$root_directory$(basename $folder)
    python3 vo_trajectory_from_folder.py  --model-name tartanvo_1914.pkl \
                                    --tum \
                                    --batch-size 1 --worker-num 1 \
                                    --test-dir  "$folder_name/rgb"\
                                    --pose-file "$folder_name/alter_groundtruth.txt" \

done
