import gc
import time
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint

import torch

from transformers import AutoTokenizer

import wandb
import os
import sys
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "true"
sys.path.append('../src')

from utils import load_filepaths
from utils import get_config, dictionary_to_namespace, get_logger, save_config, update_filepaths
from criterion.score import get_score
from data.preprocessing import make_folds, get_max_len_from_df, get_additional_special_tokens, preprocess_text
from dataset.datasets import get_train_dataloader, get_valid_dataloader
from dataset.collators import collate
from models.utils import get_model
from optimizer.optimizer import get_optimizer
from utils import AverageMeter, time_since, get_evaluation_steps
from scheduler.scheduler import get_scheduler
from adversarial_learning.awp import AWP
from criterion.criterion import get_criterion

from datetime import datetime
from utils import str_to_bool, create_dirs_if_not_exists


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--debug', type=str_to_bool, default=False)
    parser.add_argument('--use_wandb', type=str_to_bool, default=True)
    parser.add_argument('--fold', type=int)
    arguments = parser.parse_args()
    return arguments


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def valid_fn(valid_dataloader, model, criterion, epoch):
    valid_losses = AverageMeter()
    model.eval()
    predictions = []
    start = time.time()

    for step, (inputs, labels) in enumerate(valid_dataloader):
        inputs = collate(inputs)

        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = labels.to(device)

        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)

        if config.training.gradient_accumulation_steps > 1:
            loss = loss / config.training.gradient_accumulation_steps

        valid_losses.update(loss.item(), batch_size)
        predictions.append(y_preds.to('cpu').numpy())

        if step % config.general.valid_print_frequency == 0 or step == (len(valid_dataloader) - 1):
            remain = time_since(start, float(step + 1) / len(valid_dataloader))
            logger.info('EVAL: [{0}][{1}/{2}] '
                        'Elapsed: {remain:s} '
                        'Loss: {loss.avg:.4f} '
                        .format(epoch+1, step+1, len(valid_dataloader),
                                remain=remain,
                                loss=valid_losses))

        if args.use_wandb:
            wandb.log({f"Validation loss": valid_losses.val})

    predictions = np.concatenate(predictions)
    return valid_losses, predictions


def inference_fn(test_loader, model):
    predictions = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        predictions.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(predictions)
    return 0, predictions


def train_loop(train_folds,
               valid_folds,
               model_checkpoint_path=None):

    train_dataloader = get_train_dataloader(config, train_folds)
    valid_dataloader = get_valid_dataloader(config, valid_folds)

    valid_labels = valid_folds[config.general.target_columns].values

    model = get_model(config, model_checkpoint_path=model_checkpoint_path)
    torch.save(model.backbone_config, filepaths['backbone_config_fn_path'])
    model.to(device)

    optimizer = get_optimizer(model, config)

    train_steps_per_epoch = int(len(train_folds) / config.general.train_batch_size)
    num_train_steps = train_steps_per_epoch * config.training.epochs

    eval_steps = get_evaluation_steps(train_steps_per_epoch,
                                      config.training.evaluate_n_times_per_epoch)

    scheduler = get_scheduler(optimizer, config, num_train_steps)

    awp = AWP(model=model,
              optimizer=optimizer,
              adv_lr=config.adversarial_learning.adversarial_lr,
              adv_eps=config.adversarial_learning.adversarial_eps,
              adv_epoch=config.adversarial_learning.adversarial_epoch_start)

    criterion = get_criterion(config)

    best_score = np.inf
    for epoch in range(config.training.epochs):

        start_time = time.time()

        model.train()

        scaler = torch.cuda.amp.GradScaler(enabled=config.training.apex)

        train_losses = AverageMeter()
        valid_losses = None
        score, scores = None, None

        start = time.time()
        global_step = 0

        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = collate(inputs)

            for k, v in inputs.items():
                inputs[k] = v.to(device)

            labels = labels.to(device)
            awp.perturb(epoch)

            batch_size = labels.size(0)

            with torch.cuda.amp.autocast(enabled=config.training.apex):
                y_preds = model(inputs)
                loss = criterion(y_preds, labels)

            if config.training.gradient_accumulation_steps > 1:
                loss = loss / config.training.gradient_accumulation_steps

            train_losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()

            awp.restore()

            if config.training.unscale:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            if args.use_wandb:
                wandb.log({f"Training loss": train_losses.val})

            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                if config.scheduler.batch_scheduler:
                    scheduler.step()

            if (step % config.general.train_print_frequency == 0) or \
                    (step == (len(train_dataloader) - 1)) or \
                    (step + 1 in eval_steps) or \
                    (step - 1 in eval_steps):

                remain = time_since(start, float(step + 1) / len(train_dataloader))
                logger.info(f'Epoch: [{epoch+1}][{step+1}/{len(train_dataloader)}] '
                            f'Elapsed {remain:s} '
                            f'Loss: {train_losses.val:.4f}({train_losses.avg:.4f}) '
                            f'Grad: {grad_norm:.4f}  '
                            f'LR: {scheduler.get_lr()[0]:.8f}  ')

            if (step + 1) in eval_steps:
                valid_losses, predictions = valid_fn(valid_dataloader, model, criterion, epoch)
                score, scores = get_score(valid_labels, predictions)

                model.train()

                logger.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
                if score < best_score:
                    best_score = score

                    torch.save({'model': model.state_dict(), 'predictions': predictions}, filepaths['model_fn_path'])
                    logger.info(f'\nEpoch {epoch + 1} - Save Best Score: {best_score:.4f} Model\n')

                unique_parameters = ['.'.join(name.split('.')[:4]) for name, _ in model.named_parameters()]
                learning_rates = list(set(zip(unique_parameters, scheduler.get_lr())))

                if args.use_wandb:
                    wandb.log({f'{parameter} lr': lr for parameter, lr in learning_rates})
                    wandb.log({f'Best Score': best_score})

        if config.optimizer.use_swa:
            optimizer.swap_swa_sgd()

        elapsed = time.time() - start_time

        logger.info(f'Epoch {epoch + 1} - avg_train_loss: {train_losses.avg:.4f} '
                    f'avg_val_loss: {valid_losses.avg:.4f} time: {elapsed:.0f}s '
                    f'Epoch {epoch + 1} - Score: {score:.4f}  Scores: {scores}\n'
                    '=============================================================================\n')

        if args.use_wandb:
            wandb.log({f"Epoch": epoch + 1,
                       f"avg_train_loss": train_losses.avg,
                       f"avg_val_loss": valid_losses.avg,
                       f"Score": score,
                       f"Cohesion rmse": scores[0],
                       f"Syntax rmse": scores[1],
                       f"Vocabulary rmse": scores[2],
                       f"Phraseology rmse": scores[3],
                       f"Grammar rmse": scores[4],
                       f"Conventions rmse": scores[5]})

    predictions = torch.load(filepaths['model_fn_path'], map_location=torch.device('cpu'))['predictions']
    valid_folds[[f"pred_{c}" for c in config.general.target_columns]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


def get_result(oof_df):
    labels = oof_df[config.general.target_columns].values
    preds = oof_df[[f"pred_{c}" for c in config.general.target_columns]].values
    score, scores = get_score(labels, preds)
    print(f'Score: {score:<.4f}  Scores: {scores}')


def check_arguments():
    all_folds = [i for i in range(config.general.n_folds)]
    assert args.fold in all_folds, \
        f'Invalid training fold, fold number must be in {all_folds}'

    if config.general.use_current_data_pseudo_labels and config.general.use_current_data_true_labels:
        logger.warning('Both use_current_data_pseudo_labels and use_current_data_true_labels are True. ')


def init_wandb():
    job_type = 'debug' if args.debug else 'train'
    mode = 'finetuning/from checkpoint' if config['model']['from_checkpoint'] == '' else 'pretraining/from scratch'
    backbone_type = config['model']['backbone_type']
    criterion_type = config['criterion']['criterion_type']
    pooling_type = config['model']['pooling_type']

    wandb.login(key='')

    wandb_run = wandb.init(
                    project=config['logging']['wandb']['project'],
                    # group=config['model']['backbone_type'],
                    group=args.run_id,
                    job_type=job_type,
                    tags=[backbone_type, mode, job_type, 'fold'+str(args.fold),
                          criterion_type, pooling_type, args.run_id],
                    config=config,
                    name=f'{args.run_id}-fold{args.fold}'
    )
    return wandb_run


def main():
    train = pd.read_csv(filepaths['TRAIN_CSV_PATH'])

    train = make_folds(train,
                       target_cols=config.general.target_columns,
                       n_splits=config.general.n_folds,
                       random_state=config.general.seed)

    train['full_text'] = train['full_text'].apply(preprocess_text)

    special_tokens_replacement = get_additional_special_tokens()
    all_special_tokens = list(special_tokens_replacement.values())

    tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_type,
                                              use_fast=True,
                                              additional_special_tokens=all_special_tokens, )

    tokenizer.save_pretrained(filepaths['tokenizer_dir_path'])
    config.tokenizer = tokenizer

    train_df = pd.DataFrame(columns=train.columns)
    valid_df = train[train['fold'] == fold].reset_index(drop=True)

    if config.general.use_current_data_true_labels:
        train_df = pd.concat([train_df, train[train['fold'] != fold].reset_index(drop=True)], axis=0)

    if config.general.use_previous_data_pseudo_labels:
        pseudo_path = filepaths["prev_data_pseudo_fn_path"]
        logger.info(f'Loading previous data pseudo labels: {pseudo_path}')

        fold_pseudo = pd.read_csv(pseudo_path)
        fold_pseudo['in_train'] = fold_pseudo['text_id'].apply(lambda x: x in train['text_id'].values)
        fold_pseudo = fold_pseudo[~fold_pseudo['in_train'].values]
        fold_pseudo = fold_pseudo[['text_id', 'full_text'] + config.general.target_columns]

        train_df = pd.concat([train_df, fold_pseudo], axis=0).reset_index(drop=True)

    if config.general.use_current_data_pseudo_labels:
        pseudo_path = filepaths['curr_data_pseudo_fn_path']
        logger.info(f'Loading current data pseudo labels: {pseudo_path}')

        fold_pseudo = pd.read_csv(pseudo_path)
        fold_pseudo = fold_pseudo[['text_id'] + config.general.target_columns]
        fold_pseudo = pd.merge(fold_pseudo, train[['text_id', 'full_text', 'fold']], on='text_id', how='left')
        fold_pseudo = fold_pseudo[fold_pseudo['fold'] != fold].reset_index(drop=True)

        train_df = pd.concat([train_df, fold_pseudo], axis=0).reset_index(drop=True)

    train_df[config.general.target_columns] = train_df[config.general.target_columns].clip(1, 5)

    if args.debug:
        logger.info('Debug mode: using only 50 samples')
        train_df = train_df.sample(n=50, random_state=config.general.seed).reset_index(drop=True)
        valid_df = valid_df.sample(n=50, random_state=config.general.seed).reset_index(drop=True)

    logger.info(f'Train shape: {train_df.shape}')
    logger.info(f'Valid shape: {valid_df.shape}')

    if config.general.set_max_length_from_data:
        logger.info('Setting max length from data')
        config.general.max_length = get_max_len_from_df(train_df, tokenizer)

    logger.info(f"Max tokenized sequence len: {config.general.max_length}")
    logger.info(f"==================== fold: {fold} training ====================")

    model_checkpoint_path = filepaths['model_checkpoint_fn_path'] if config.model.from_checkpoint else None
    logger.info(f'Using model checkpoint from: {model_checkpoint_path}')

    if args.debug:
        config.training.epochs = 1

    if config.general.check_cv_on_all_data and not args.debug:
        '''
            For models 15 and 16 pretraining step I checked CV on all available data, instead of 1 folds
            Its results in overfitting, and I changed it later, but keep it here to reproduce results,
            since I used pseudolabels from model 16 to train model 17, so without this 
        '''
        fold_out = train_loop(train_df,
                              train,
                              model_checkpoint_path=model_checkpoint_path, )
    else:
        fold_out = train_loop(train_df,
                              valid_df,
                              model_checkpoint_path=model_checkpoint_path,)

    fold_out.to_csv(filepaths['oof_fn_path'], index=False)

    wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    filepaths = load_filepaths()

    config_path = os.path.join(filepaths['CONFIGS_DIR_PATH'], args.config_name)
    config = get_config(config_path)

    fold = args.fold
    if args.use_wandb:
        run = init_wandb()

    filepaths = update_filepaths(filepaths, config, args.run_id, fold)
    create_dirs_if_not_exists(filepaths)

    if not os.path.exists(filepaths['run_dir_path']):
        os.makedirs(filepaths['run_dir_path'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger(filename=filepaths['log_fn_path'])

    if os.path.isfile(filepaths['model_fn_path']):
        new_name = filepaths["model_fn_path"]+f'_renamed_at_{str(datetime.now())}'
        logger.warning(f'{filepaths["model_fn_path"]} is already exists, renaming this file to {new_name}')
        os.rename(filepaths["model_fn_path"], new_name)

    # save_config(config, filepaths['training_config_fn_path'])

    config = dictionary_to_namespace(config)

    seed_everything(seed=config.general.seed)

    check_arguments()
    main()
