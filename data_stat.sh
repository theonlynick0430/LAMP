#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <min length(# imgs) of trajs [ONLY USED FOR PART OF THE OUTPUT]>"
    exit 1
fi

l=$1

max_trajs=-1
min_trajs=-1
trajs_with_at_least_l=0
total_imgs=0
total_trajs=0

# process substitution to avoid modifying vars in subshell
while IFS= read -r subfolder; do
    if [ -d "$subfolder/images0" ] && [ -f "$subfolder/lang.txt" ]; then # img folder and lang label exists
        file_count=$(find "$subfolder/images0" -maxdepth 1 -type f | wc -l)

        # update max_trajs and min_trajs
        if [ "$max_trajs" -eq -1 ] || [ "$file_count" -gt "$max_trajs" ]; then
            max_trajs=$file_count
        fi

        if [ "$min_trajs" -eq -1 ] || [ "$file_count" -lt "$min_trajs" ]; then
            min_trajs=$file_count
        fi

        # update trajs_with_at_least_l
        if [ "$file_count" -ge "$l" ]; then
            ((trajs_with_at_least_l++))
        fi

        # update vars for mean calculation
        ((total_imgs += file_count))
        ((total_trajs += 1))
    fi
done < <(find /nfs/turbo/coe-jjparkcv/datasets/bridge/raw/bridge_data_v2 -type d -name "traj[0-9]*")

# mean calc
if [ "$total_trajs" -gt 0 ]; then
    mean_imgs=$((total_imgs / total_trajs))
else
    mean_imgs=0
fi

echo "min # of imgs in a traj: $min_trajs"
echo "max # of imgs in a traj: $max_trajs"
echo "# of trajs containing at least $l files: $subfolders_with_at_least_l"
echo "mean # of imgs in trajs: $mean_imgs"
