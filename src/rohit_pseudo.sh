#!/bin/bash

set -ex

python make_rohit_pseudolabels.py --model_id exp01_fb3
python make_rohit_pseudolabels.py --model_id exp01_fb3_part2
python make_rohit_pseudolabels.py --model_id exp02_fb3
python make_rohit_pseudolabels.py --model_id exp02_fb3_part2_distilbart
python make_rohit_pseudolabels.py --model_id exp02_fb3_part2_distilbert
python make_rohit_pseudolabels.py --model_id exp03_fb3
python make_rohit_pseudolabels.py --model_id exp04_fb3
python make_rohit_pseudolabels.py --model_id exp12_fb3
python make_rohit_pseudolabels.py --model_id exp13_fb3
python make_rohit_pseudolabels.py --model_id exp14_fb3_deberta
python make_rohit_pseudolabels.py --model_id exp14_fb3_roberta
