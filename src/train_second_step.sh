#!/bin/bash

set -ex

## Rohit pseudolabels here

python make_pseudolabels_ensemble.py --ensemble_id ensemble_4419


for model_id in model52
do
  python train.py --config_name "${model_id}_pretraining_training_config.yaml" --run_id "${model_id}_pretrain" --debug False --use_wand False --fold 0
  python train.py --config_name "${model_id}_pretraining_training_config.yaml" --run_id "${model_id}_pretrain" --debug False --use_wand False --fold 1
  python train.py --config_name "${model_id}_pretraining_training_config.yaml" --run_id "${model_id}_pretrain" --debug False --use_wand False --fold 2
  python train.py --config_name "${model_id}_pretraining_training_config.yaml" --run_id "${model_id}_pretrain" --debug False --use_wand False --fold 3
  python train.py --config_name "${model_id}_pretraining_training_config.yaml" --run_id "${model_id}_pretrain" --debug False --use_wand False --fold 4

  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 0 
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 1 
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 2 
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 3 
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 4 
  
  python inference.py --model_dir_path "../models/${model_id}" --mode oofs --debug False
  python inference.py --model_dir_path "../models/${model_id}" --mode prev_pseudolabels --debug False
done


for model_id in model67 model68 model69 model70
do
  
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 0
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 1
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 2
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 3
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 4
  
  python inference.py --model_dir_path "../models/${model_id}" --mode oofs --debug False
  python inference.py --model_dir_path "../models/${model_id}" --mode prev_pseudolabels --debug False
done


for model_id in model21 model23 model30 model31 model37 model43 model45 model52 model67 model68 model69 model70
do
  python inference.py --model_dir_path "../models/${model_id}" --mode curr_pseudolabels --debug --True
done
python make_pseudolabels_ensemble.py --ensemble_id current_train_ensemble_4434


for model_id in model71 model75 model81 model84
do

  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 0
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 1
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 2
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 3
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 4

  python inference.py --model_dir_path "../models/${model_id}" --mode oofs --debug False
  python inference.py --model_dir_path "../models/${model_id}" --mode prev_pseudolabels --debug False
done
