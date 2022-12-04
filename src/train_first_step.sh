#!/bin/bash

set -ex


for model_id in model2
do
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 0
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 1
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 2
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 3
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 4
  
  python inference.py --model_dir_path "../models/${model_id}" --mode oofs --debug False
  python inference.py --model_dir_path "../models/${model_id}" --mode prev_pseudolabels --debug False
done


for model_id in model15 model16 model17 model18 model19 model20
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
python make_pseudolabels_ensemble.py --ensemble_id ensemble_44631


for model_id in model21
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
python make_pseudolabels_ensemble.py --ensemble_id ensemble_4459


for model_id in model22
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

python make_model23_pretrain.py

for model_id in model23
do

  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 0
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 1
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 2
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 3
  python train.py --config_name "${model_id}_training_config.yaml" --run_id $model_id --debug False --use_wand False --fold 4

  python inference.py --model_dir_path "../models/${model_id}" --mode oofs --debug False
  python inference.py --model_dir_path "../models/${model_id}" --mode prev_pseudolabels --debug False
done


for model_id in model24 model26 model27 model28 model29
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

python make_pseudolabels_ensemble.py --ensemble_id ensemble_4449


for model_id in model30
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
python make_pseudolabels_ensemble.py --ensemble_id ensemble_4448


for model_id in model31 model32
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
python make_pseudolabels_ensemble.py --ensemble_id ensemble_4447


for model_id in model33 model34 model35
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
python make_pseudolabels_ensemble.py --ensemble_id ensemble_44439


for model_id in model36 model37
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
python make_pseudolabels_ensemble.py --ensemble_id ensemble_444371


for model_id in model38 model39 model40 model41 model42 model43 model44
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
python make_pseudolabels_ensemble.py --ensemble_id ensemble_444335


for model_id in model45 model46 model47 model48 model49 model50 model51
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
