#!/bin/bash
# Created on 2020/07
# Author: Nabil Sarwar

python_path=/Work19/2019/nabil/envs/crnn/bin/python
src=/Work19/2019/nabil/ecctn_1
stage=11

if [ $stage -le 11 ]; then
  echo "Stage 1: Training"
  ${python_path} ${src}/train.py \
    --config ${src}/config/ecctn_1.json \
    --device 1
fi

if [ $stage -le 0 ]; then
  echo "Stage 2: Enhancing"
  ${python_path} ${src}/enhancement.py \
    --config ${src}/config/test_config.json
fi

if [ $stage -le 0 ]; then
    echo "Stage 3: Evaluation."
    models=ecctn_1
    dirs=/Work19/2019/nabil/ecctn_1/exp/
    /opt18/matlab_2015b/bin/matlab -nodesktop -nosplash -r "models='$models';dirs='$dirs';eval;quit"
fi