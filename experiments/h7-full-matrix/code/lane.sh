#!/bin/bash
# $1 = lane index, $2 = physical GPU, $3 = number of lanes
cd /data/cl195/NeurIPS_workshop; source env.sh
j=0
while read task model; do
  if [ $((j % $3)) -eq $1 ]; then
    out="experiments/h7-full-matrix/results/${task}_${model}.jsonl"
    if [ -s "$out" ]; then echo "skip $task $model (exists)"; else
      echo "=== $task $model on GPU $2 ==="
      CUDA_VISIBLE_DEVICES=$2 $SC_PY experiments/h7-full-matrix/code/run_matrix.py \
        --model "$model" --task "$task" --device cuda:0 2>&1 | grep -v "it/s\|Loading weights"
    fi
  fi
  j=$((j+1))
done < experiments/h7-full-matrix/code/jobs2.txt
echo "LANE $1 DONE"
