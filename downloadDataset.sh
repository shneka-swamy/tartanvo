# Download the tar file from the website

downloadDir=./tumvi/exported/euroc/512_16

list_dataset=(dataset-corridor1_512_16.tar dataset-corridor2_512_16.tar dataset-corridor3_512_16.tar 
                 dataset-corridor4_512_16.tar dataset-corridor5_512_16.tar
                 dataset-magistrale1_512_16.tar dataset-magistrale2_512_16.tar dataset-magistrale3_512_16.tar
                 dataset-magistrale4_512_16.tar dataset-magistrale5_512_16.tar dataset-magistrale6_512_16.tar
                 dataset-outdoors1_512_16.tar dataset-outdoors2_512_16.tar dataset-outdoors3_512_16.tar 
                 dataset-outdoors4_512_16.tar dataset-outdoors5_512_16.tar dataset-outdoors6_512_16.tar
                 dataset-outdoors7_512_16.tar dataset-outdoors8_512_16.tar
                 dataset-room1_512_16.tar dataset-room2_512_16.tar dataset-room3_512_16.tar
                 dataset-room4_512_16.tar dataset-room5_512_16.tar dataset-room6_512_16.tar
                 dataset-slides1_512_16.tar dataset-slides2_512_16.tar dataset-slides3_512_16.tar
                )

mkdir -p $downloadDir

pushd $downloadDir

for i in "${list_dataset[@]}"
do
    url=https://cdn2.vision.in.tum.de/tumvi/exported/euroc/512_16/$i
    if [ ! -f $i.md5 ]; then
        echo "Downloading $i.md5"
        wget $url.md5 > /dev/null 2>&1
    fi
    # check checksum or if file does not exist
    if [ ! -f $i ] || ! md5sum -c $i.md5; then
        rm -f $i
        rm -f $i.md5
        echo "Downloading $i.md5 again"
        wget $url.md5 > /dev/null 2>&1
        echo "Downloading $i"
        wget $url > /dev/null 2>&1
        md5sum -c $i.md5
        # if checksum still fails, exit
        if [ $? -ne 0 ]; then
            echo "Checksum failed ${i} , with checksum file ${i}.md5"
            exit 1
        fi
    fi

    tar -xvf $i
done

popd
