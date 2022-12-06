#!/bin/bash
scan_ids=(0 1 2 3 4 5 6 7 8 9 10 11 12)
for scan_id in ${scan_ids[@]}; do
    python training/exp_runner.py --scan_id $scan_id --local_rank 0 --conf confs/eth3d_freqs_2d.conf
done