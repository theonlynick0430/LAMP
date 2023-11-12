#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <num train trajs> <num test trajs> <min length of traj>"
    exit 1
fi

# clean existing symlinks
find ./bridge -type l -delete
if test -d "./bridge"; then
    rm -r ./bridge
fi

# num train trajs
num_train_traj=$1
# num test trajs
num_test_traj=$2
# min length of traj
l=$3

num_traj=$((num_train_traj + num_test_traj))
train=true
count=0

mkdir ./bridge
mkdir ./bridge/train
mkdir ./bridge/test
# process subsitution to avoid modifying vars in subshell
while IFS= read -r subfolder; do
    if [ -d "$subfolder/images0" ] && [ -f "$subfolder/lang.txt" ]; then # img folder and lang label exists
        file_count=$(find "$subfolder/images0" -maxdepth 1 -type f | wc -l)
        if [ "$file_count" -ge "$l" ]; then # min length of traj satisfied
            # done generating trajs
            [ "$count" -eq "$num_traj" ] && break
            # done generating train trajs
            if [ "$count" -eq "$num_train_traj" ]; then
                train=false
            fi
            if [ "$train" = true ]; then
                # create a symlink for train data
                ln -s "$subfolder" "./bridge/train/traj$count"
            else
                # create a symlink for test data
                ln -s "$subfolder" "./bridge/test/traj$((count-num_train_traj))"
            fi
            ((count++))
        fi
    fi
done < <(find /nfs/turbo/coe-jjparkcv/datasets/bridge/raw/bridge_data_v2 -type d -name "traj[0-9]*")

echo "dataset with $num_train_traj train and $num_test_traj test trajs configured at ./bridge"
