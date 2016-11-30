#!/bin/bash

function run_lr {
  mcs=$1
  scale=$2
  echo -n "Cores $mcs "
  /root/spark/bin/spark-submit --total-executor-cores $mcs ./mllib_lr.py $scale $mcs 2>&1 | grep "LR.*took"
}

run_lr 32  0.125000
run_lr 4  0.015625
run_lr 4  0.021382
run_lr 12  0.050164
run_lr 12  0.055921
run_lr 12  0.061678
run_lr 16  0.061678
