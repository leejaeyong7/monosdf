SCANNET_SCENES=(1 2 3 4)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/scannet_grids.conf --scan_id 1
for scene in "${SCANNET_SCENES[@]}" 
do
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/scannet_qff.conf --scan_id $scene
done


DTU_SCENES=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)

for scene in "${DTU_SCENES[@]}" 
do
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/dtu_qff_3views.conf --scan_id $scene
done

