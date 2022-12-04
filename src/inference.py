import os
import gc
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import torch

from transformers import AutoTokenizer
from dataset.datasets import get_test_dataloader
from data.preprocessing import get_max_len_from_df, make_folds, preprocess_text
from utils import get_config, load_filepaths
from models.utils import get_model
from dataset.collators import collate
from utils import dictionary_to_namespace

import argparse
from utils import str_to_bool

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir_path', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--debug', type=str_to_bool, default=False)
    arguments = parser.parse_args()
    return arguments


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    model_id = args.model_dir_path.split("/")[-1]
    filepaths = load_filepaths()

    config_path = os.path.join(filepaths['CONFIGS_DIR_PATH'], f'{model_id}_training_config.yaml')
    config = get_config(config_path)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_dir_path, 'tokenizer/'))

    config = dictionary_to_namespace(config)
    config.tokenizer = tokenizer
    config.model.pretrained_backbone = False

    assert args.mode in ['prev_pseudolabels', 'curr_pseudolabels', 'submission', 'oofs']

    seed_everything(seed=config.general.seed)

    dataframe_path = None
    if args.mode == 'submission':
        dataframe_path = filepaths['TEST_CSV_PATH']
    elif args.mode == 'oofs':
        dataframe_path = filepaths['TRAIN_CSV_PATH']
    elif args.mode == 'prev_pseudolabels':
        dataframe_path = filepaths['PREVIOUS_DATA_CSV_PATH']
    elif args.mode == 'curr_pseudolabels':
        dataframe_path = filepaths['TRAIN_CSV_PATH']

    test_df = pd.read_csv(dataframe_path)

    if args.mode == 'oofs':
        test_df = make_folds(test_df,
                             target_cols=config.general.target_columns,
                             n_splits=config.general.n_folds,
                             random_state=config.general.seed)

    elif args.mode == 'prev_pseudolabels':
        df = pd.read_csv(filepaths['TRAIN_CSV_PATH'])
        test_df['in_fb3'] = test_df['text_id'].apply(lambda x: x in df.text_id.values)
        test_df = test_df[~test_df['in_fb3'].values]

    if args.debug:
        test_df = test_df.sample(50, random_state=42)

    test_df['full_text'] = test_df['full_text'].apply(preprocess_text)

    test_df['tokenize_length'] = [len(config.tokenizer(text)['input_ids']) for text in test_df['full_text'].values]
    test_df = test_df.sort_values('tokenize_length', ascending=True).reset_index(drop=True)

    if config.general.set_max_length_from_data:
        config.general.max_length = get_max_len_from_df(test_df, config.tokenizer)

    if args.debug:
        test_df = test_df.sample(50, random_state=1)

    target_columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

    predictions = []
    oofs_list = []
    for fold in range(config.general.n_folds):

        subset = test_df.copy()
        if args.mode == 'oofs':
            subset = subset[subset['fold'] == fold].reset_index(drop=True)

        test_dataloader = get_test_dataloader(config, subset)

        backbone_type = config.model.backbone_type.replace('/', '-')
        model_checkpoint_path = os.path.join(args.model_dir_path, f"{backbone_type}_fold{fold}_best.pth")
        backbone_config_path = os.path.join(args.model_dir_path, 'config.pth')

        model = get_model(config,
                          backbone_config_path=backbone_config_path,
                          model_checkpoint_path=model_checkpoint_path,
                          train=False)

        prediction = inference_fn(test_dataloader, model, device)
        predictions.append(prediction)

        gc.collect()
        torch.cuda.empty_cache()

        if args.mode in ['prev_pseudolabels', 'curr_pseudolabels', 'oofs']:
            out = pd.DataFrame(prediction, columns=['cohesion', 'syntax', 'vocabulary',
                                                    'phraseology', 'grammar', 'conventions'])

            for col in ['text_id', 'full_text', 'tokenize_length', 'in_fb3']:
                if col in subset.columns:
                    out[col] = subset[col]

            if args.mode in ['prev_pseudolabels', 'curr_pseudolabels']:
                pseudo_path = filepaths['PREVIOUS_DATA_PSEUDOLABELS_DIR_PATH'] \
                    if args.mode == 'prev_pseudolabels' \
                    else filepaths['CURRENT_DATA_PSEUDOLABELS_DIR_PATH']

                dir_path = os.path.join(pseudo_path, f'{model_id}_pseudolabels')
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)

                out.to_csv(os.path.join(dir_path, f'pseudolabels_fold{fold}.csv'), index=False)

            if args.mode == 'oofs':
                oofs_list.append(out)

        del model, prediction

    if args.mode == 'submission':
        predictions = np.mean(predictions, axis=0)

        test_df[target_columns] = predictions

        submission = pd.read_csv(filepaths['SAMPLE_SUBMISSION_CSV_PATH'])
        submission = submission.drop(columns=target_columns) \
            .merge(test_df[['text_id'] + target_columns], on='text_id', how='left')
        submission[['text_id'] + target_columns].to_csv(os.path.join(filepaths['SUBMISSIONS_DIR_PATH'], f'{model_id}_submission.csv'), index=False)

    elif args.mode == 'oofs':
        oof_df = pd.concat(oofs_list)
        oof_df.to_csv(os.path.join(filepaths['OOFS_DIR_PATH'], f'{model_id}.csv'), index=False)
