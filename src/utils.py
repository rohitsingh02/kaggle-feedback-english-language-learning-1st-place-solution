import sys
import yaml
from types import SimpleNamespace
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

import os
import time
import math
import json


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (remain %s)' % (as_minutes(s), as_minutes(rs))


def get_evaluation_steps(num_train_steps, n_evaluations):
    eval_steps = num_train_steps // n_evaluations
    eval_steps = [eval_steps * i for i in range(1, n_evaluations + 1)]
    return eval_steps


def get_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, path):
    with open(path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def load_filepaths():
    with open('../SETTINGS.json') as f:
        filepaths = json.load(f)

    for key, value in filepaths.items():
        filepaths[key] = os.path.abspath(value)
    return filepaths


def update_filepaths(filepaths, config, run_name, fold):

    backbone_type = config['model']['backbone_type'].replace('/', '-')
    model_fn = f"{backbone_type}_fold{fold}_best.pth"
    pseudolabels_fn = f'pseudolabels_fold{fold}.csv'

    filepaths['run_dir_path'] = os.path.join(filepaths['MODELS_DIR_PATH'], run_name)
    filepaths['model_fn_path'] = os.path.join(filepaths['run_dir_path'], model_fn)
    filepaths['backbone_config_fn_path'] = os.path.join(filepaths['run_dir_path'], 'config.pth')
    filepaths['tokenizer_dir_path'] = os.path.join(filepaths['run_dir_path'], 'tokenizer')
    filepaths['training_config_fn_path'] = os.path.join(filepaths['CONFIGS_DIR_PATH'], f'{run_name}_training_config.yaml')
    filepaths['log_fn_path'] = os.path.join(filepaths['run_dir_path'], 'train.log')
    filepaths['oof_fn_path'] = os.path.join(filepaths['run_dir_path'], f'oof_fold{fold}.csv')

    filepaths['prev_data_pseudo_fn_path'] = os.path.join(filepaths['PREVIOUS_DATA_PSEUDOLABELS_DIR_PATH'],
                                                         config['general']['previous_data_pseudo_version'],
                                                         pseudolabels_fn)

    filepaths['curr_data_pseudo_fn_path'] = os.path.join(filepaths['CURRENT_DATA_PSEUDOLABELS_DIR_PATH'],
                                                         config['general']['current_data_pseudo_version'],
                                                         pseudolabels_fn)

    filepaths['model_checkpoint_fn_path'] = os.path.join(filepaths['MODELS_DIR_PATH'],
                                                         config['model']['checkpoint_id'],
                                                         model_fn) \
        if config['model']['from_checkpoint'] else ''

    for key, value in filepaths.items():
        filepaths[key] = os.path.abspath(value)
    return filepaths


def dictionary_to_namespace(data):
    if type(data) is list:
        return list(map(dictionary_to_namespace, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, dictionary_to_namespace(value))
        return sns
    else:
        return data


def get_logger(filename):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=filename)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def str_to_bool(argument):
    if isinstance(argument, bool):
        return argument
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_dirs_if_not_exists(filepaths):
    for key, value in filepaths.items():
        if 'DIR_PATH' in key.upper() and not os.path.isdir(value):
            os.mkdir(value)
