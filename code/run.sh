#!/bin/bash
python training/exp_runner.py --scan_id 24 --local_rank 0 --conf confs/dtu_freqs_2d_fullres_allviews.conf 
python training/exp_runner.py --scan_id 24 --local_rank 0 --conf confs/dtu_freqs_fullres_allviews.conf 
python training/exp_runner.py --scan_id 24 --local_rank 0 --conf confs/dtu_mlp_fullres_allviews.conf 
# python training/exp_runner.py --scan_id 24 --local_rank 0 --conf confs/dtu_grids_fullres_allviews.conf 