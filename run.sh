#!/bin/bash
MOUNT_DIR="/home/lee896/data/$(hostname -s)"
DATA_DIR="${MOUNT_DIR}/monosdf"
EXP_DIR="${MOUNT_DIR}/monosdf-qff-results"
run_name=""
scene=$1
qff_type=$2
dtu=$3
if [[ $2 = "0" ]]; then
  model="mlp"
elif [[ $2 = "4" ]]; then
  model="grids"
else
  model="qff_${qff_type}"
fi

if [[ $3 = "--dtu" ]]; then
  dataset="dtu"
  suffix="_3views"
else
  dataset="scannet"
  suffix=""
fi
conf=${dataset}_${model}${suffix}.conf
run_name=${dataset}_${model}${suffix}
echo $run_name


cd code
python training/exp_runner.py --conf confs/${conf} --scan_id $scene --data_folder=${DATA_DIR} --exps_folder=${EXP_DIR} --timestamp=server

# zip files
cd ${EXP_DIR}
zip -r "${run_name}_${scene}.zip" "${run_name}_${scene}/server/surfaces"


result_dirname="monosdf-qff-rerun"
ssh jyl "mkdir -p /mnt/data1/cluster_results/${result_dirname}/${run_name}"
ssh jyl "mkdir -p /mnt/data1/monosdf_outputs/${result_dirname}/${run_name}/${scene}"
scp "${run_name}_${scene}.zip" jyl:/mnt/data1/cluster_results/${result_dirname}/${run_name}/
# ssh jyl "cd /mnt/data1/cluster_results/${result_dirname}/${run_name}/ && unzip "{run_name}_${scene}.zip" -d /mnt/data1/monosdf_outputs/${result_dirname}/${run_name}/${scene}"
ssh jyl "cd /mnt/data1/cluster_results/${result_dirname}/${run_name}/ && unzip "{run_name}_${scene}.zip" -d /mnt/data1/monoqff/outputs/${run_name}_${scene}"

