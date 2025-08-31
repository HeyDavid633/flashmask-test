#!/bin/bash

batch_seq_pairs=("1 128" "8 512" "16 2048")

for bs_seq in "${batch_seq_pairs[@]}"; do
    bs=$(echo $bs_seq | cut -d' ' -f1)
    seq=$(echo $bs_seq | cut -d' ' -f2)

            script -a -c "python benchmk_end2end.py \
                --batch_size=$bs \
                --seq_len=$seq" \
                 "/root/fusion-SC25/SC25-STOF-AD/src/benchmk_end2end.txt"

    done
    



