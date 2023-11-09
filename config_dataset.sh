#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <num trajs> <min length(# imgs) of traj>"
    exit 1
fi

# clean existing symlinks
find ./bridge -type l -delete
if test -d "./bridge"; then
    rm -r ./bridge
fi

# num trajs
n=$1
# min length of traj
l=$2

count=0

mkdir ./bridge
# process subsitution to avoid modifying vars in subshell
while IFS= read -r subfolder; do
    if [ -d "$subfolder/images0" ] && [ -f "$subfolder/lang.txt" ]; then # img folder and lang label exists
        file_count=$(find "$subfolder/images0" -maxdepth 1 -type f | wc -l)
        if [ "$file_count" -ge "$l" ]; then # min length of traj satisfied
            # create a symlink
            ln -s "$subfolder" "./bridge/traj$count"
            ((count++))
            # break if counter reaches n
            [ "$count" -eq "$n" ] && break
        fi
    fi
done < <(find /nfs/turbo/coe-jjparkcv/datasets/bridge/raw/bridge_data_v2 -type d -name "traj[0-9]*")

echo "dataset with $count trajs configured at ./bridge"
