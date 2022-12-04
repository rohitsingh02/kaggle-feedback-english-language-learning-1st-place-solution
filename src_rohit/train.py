import time
# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from awp import AWP
# Utils
import re
from tqdm import tqdm
from collections import defaultdict

# For splitting data

from transformers import AutoTokenizer

# For Transformer Models
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset


# Suppress warnings
import warnings
warnings.filterwarnings("ignore")



import os
import numpy as np
import pandas as pd
import importlib
import sys
import time
from tqdm import tqdm
import gc
import argparse
import torch
import wandb
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from transformers import AutoTokenizer

from torch.utils.data import  DataLoader
import yaml
from types import SimpleNamespace
import os
import utils
from asyncio.log import logger


# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

sys.path.append("models")
sys.path.append("datasets")



import codecs
from text_unidecode import unidecode
from typing import Tuple

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def replace_newline(text):
    text = text.replace('\n', '[BR]')
    return text

placeholders_replacements = {
    'Generic_School': '[GENERIC_SCHOOL]',
    'Generic_school': '[GENERIC_SCHOOL]',
    'SCHOOL_NAME': '[SCHOOL_NAME]',
    'STUDENT_NAME': '[STUDENT_NAME]',
    'Generic_Name': '[GENERIC_NAME]',
    'Genric_Name': '[GENERIC_NAME]',
    'Generic_City': '[GENERIC_CITY]',
    'LOCATION_NAME': '[LOCATION_NAME]',
    'HOTEL_NAME': '[HOTEL_NAME]',
    'LANGUAGE_NAME': '[LANGUAGE_NAME]',
    'PROPER_NAME': '[PROPER_NAME]',
    'OTHER_NAME': '[OTHER_NAME]',
    'PROEPR_NAME': '[PROPER_NAME]',
    'RESTAURANT_NAME': '[RESTAURANT_NAME]',
    'STORE_NAME': '[STORE_NAME]',
    'TEACHER_NAME': '[TEACHER_NAME]',
}

def replace_placeholders(text):
    for key, value in placeholders_replacements.items():
        text = text.replace(key, value)
        
    return text


def pad_punctuation(text):
    text = re.sub('([.,!?()-])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    return text


def preprocess_text(text):
    text = resolve_encodings_and_normalize(text)
    # text = replace_newline(text) 
    text = replace_placeholders(text)
#     text = pad_punctuation(text)
    return text


def reverse_text(text):
    
    s = text.split()[::-1]
    s = " ".join(s)
    return s

# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=cfg.dataset.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[cfg.dataset.target_cols].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs



def get_scheduler(cfg, optimizer, total_steps):
    if cfg.scheduler.schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                cfg.training.warmup_epochs * (total_steps // cfg.training.batch_size)
            ),
            num_training_steps=cfg.training.epochs * (total_steps // cfg.training.batch_size),
        )
    elif cfg.scheduler.schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg.training.warmup_epochs * (total_steps // cfg.training.batch_size)),
            # num_warmup_steps=50,
            num_training_steps=cfg.training.epochs * (total_steps // cfg.training.batch_size),
            num_cycles=cfg.scheduler.num_cycles
        )
    return scheduler


def get_deberta_llrd_optimizer(cfg, model):
    
    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 
        
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = 2.0e-5
    head_lr = 5.e-4
    lr = init_lr
    
    # === Pooler and regressor ======================================================  
    
    params_0 = [p for n,p in named_parameters if ("head" in n ) 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if ("head" in n )
                and not any(nd in n for nd in no_decay)]
    

    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
    opt_parameters.append(head_params)
        
    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}    
    opt_parameters.append(head_params)
                
    # === 12 Hidden layers ==========================================================
    
    for layer in range(11,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        
        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   
                            
        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)       
        
        # lr *= 0.9
        lr *= 0.9
        
    # === Embeddings layer ==========================================================
    
    params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)
        
    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
    opt_parameters.append(embed_params)        
    
    return torch.optim.AdamW(
        opt_parameters,
        lr=init_lr,
        # lr=cfg.training.learning_rate,
        # weight_decay=cfg.optimizer.weight_decay,
        # eps=cfg.optimizer.eps,
        # betas=(0.9, 0.999)
    )



def get_optimizer(cfg, model): # best

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    differential_layers = cfg.training.differential_learning_rate_layers


    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
                # "weight_decay": 0.1,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.learning_rate,
                "weight_decay": 0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
                # "weight_decay": 0.01,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": 0,
            },
        ],
        lr=cfg.training.learning_rate,
        # weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.eps,
        betas=(0.9, 0.999)
    )

    return optimizer



def get_optimizer_grouped_parameters(
    cfg, model
):
    
    model_type="model"
    learning_rate = 1.0e-5
    weight_decay=0.01
    layerwise_learning_rate_decay = 0.9 # 0.8 train2
    
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "head" in n or "pooling" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    
    # initialize lrs for every layer
    num_layers = model.config.num_hidden_layers
    layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
        
    adam_epsilon = 1e-6 # 5e-6 train.sh
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_epsilon,
    )
        
    return optimizer




def valid_fn(valid_loader, model, criterion, device):
    losses = utils.AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, data_dict in enumerate(valid_loader):
        inputs = collate(data_dict)
        inputs = cfg.CustomDataset.batch_to_device(inputs, device)
        batch_size = data_dict['target'].size(0)
        
        with torch.no_grad():
            output_dict = model(inputs)
            loss = output_dict["loss"]

        if cfg.training.grad_accumulation > 1:
            loss = loss / cfg.training.grad_accumulation
        losses.update(loss.item(), batch_size)
        preds.append(output_dict["logits"].detach().cpu().numpy())
        end = time.time()
        if step % cfg.training.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=utils.timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions


# ====================================================
# train loop
# ====================================================

def load_checkpoint(cfg, model, fold=0):

    weight = f"{cfg.architecture.pretrained_weights}/checkpoint_{fold}.pth"

    d =  torch.load(weight, map_location="cpu")
    # print(d)
    if "model" in d:
        model_weights = d["model"]
    else:
        model_weights = d

    # if (
    #     model.embeddings.word_embeddings.weight.shape[0]
    #     < model_weights["model.embeddings.word_embeddings.weight"].shape[0]
    # ):
    #     print("resizing pretrained embedding weights")
    #     model_weights["model.embeddings.word_embeddings.weight"] = model_weights[
    #         "model.embeddings.word_embeddings.weight"
    #     ][: model.embeddings.word_embeddings.weight.shape[0]]

    try:
        model.load_state_dict(model_weights, strict=True)
    except Exception as e:
        print("removing unused pretrained layers")
        for layer_name in re.findall("size mismatch for (.*?):", str(e)):
            model_weights.pop(layer_name, None)
        model.load_state_dict(model_weights, strict=False)

    print(f"Weights loaded from: {cfg.architecture.pretrained_weights}")



def train_loop(df, fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    if "fold" in df.columns:
        val_df = df[df.fold == fold].copy()
        train_df = df[df.fold != fold].copy()
    else:
        val_df = df.copy()
        

    df_pl = pd.read_csv(f"../data/fb3_pl_fold{fold}.csv")
    
    if cfg.dataset.pretrain_pl: train_df = df_pl
    
    if cfg.dataset.use_pl:
        train_df = pd.concat([train_df, df_pl], ignore_index=True)
       
       
    if cfg.debug: 
        print("DEBUG MODE")
        train_df = train_df.head(50)
       
    print(train_df.shape) 

    # ====================================================
    # loader
    # ====================================================
    valid_labels = val_df[cfg.dataset.target_cols].values

    train_dataset = cfg.CustomDataset(train_df, mode="train", cfg=cfg)
    valid_dataset = cfg.CustomDataset(val_df, mode="val", cfg=cfg)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.training.batch_size,
                              shuffle=True,
                              num_workers=cfg.environment.num_workers, pin_memory=True, drop_last=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.training.batch_size * 2,
                              shuffle=False,
                              num_workers=cfg.environment.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer change here
    # ====================================================

    model = get_model(cfg)
    model.to(device)

    if hasattr(cfg.architecture, "pretrained_weights") and cfg.architecture.pretrained_weights != "":
        print("START LAODING PRETRAIED WEIGHTS.....")
        try:
            load_checkpoint(cfg, model, fold)
        except:
            print("WARNING: could not load checkpoint")


    if hasattr(cfg.architecture, "save_name"):
        torch.save(model.config, f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.save_name.replace('/', '-')}/config.pth")
    else:
        torch.save(model.config, f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}/config.pth")

    model.to(cfg.device)

    total_steps = len(train_dataset)
    # num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
     
    if hasattr(cfg.optimizer, "llrd") and cfg.optimizer.llrd:
        # optimizer = get_deberta_llrd_optimizer(cfg, model)
        print("Learning Rate Decay")
        optimizer = get_optimizer_grouped_parameters(cfg, model)
        
    else:
        optimizer = get_optimizer(cfg, model) 


    if hasattr(cfg.training, "priorWD") and cfg.training.priorWD:
        optimizer = utils.PriorWD(optimizer, use_prior_wd=True)


    scheduler = get_scheduler(cfg, optimizer, total_steps)
    

    # ====================================================
    # Setting Up AWP Training
    # ====================================================
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.environment.mixed_precision)
    awp = AWP(model,
        optimizer,
        adv_lr=0.0001,
        adv_eps=0.001,
        start_epoch= 2, #(cfg.training.epochs * (total_steps // cfg.training.batch_size))/cfg.training.epochs,
        scaler=scaler
    )

    criterion = nn.SmoothL1Loss(reduction='mean') # RMSELoss(reduction="mean")
    best_score = np.inf

    step_val = 0
    for epoch in range(cfg.training.epochs):
        cfg.epoch = epoch
        start_time = time.time()
        
        model.train()
        losses = utils.AverageMeter()
        start = end = time.time()
        global_step = 0
        for step, data_dict in enumerate(train_loader):
            if step_val: step_val += 1
            inputs = collate(data_dict)
            inputs = cfg.CustomDataset.batch_to_device(inputs, device)
            batch_size = data_dict['target'].size(0)
            with torch.cuda.amp.autocast(enabled=cfg.environment.mixed_precision):
                output_dict = model(inputs)
                loss = output_dict["loss"]

            if hasattr(cfg, "awp") and cfg.awp.enable and epoch >= cfg.awp.start_epoch:
                awp.attack_backward(inputs, step_val)


            if cfg.training.grad_accumulation > 1:
                loss = loss / cfg.training.grad_accumulation
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            if (step + 1) % cfg.training.grad_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                if cfg.training.batch_scheduler:
                    scheduler.step()
            end = time.time()
            if  step % cfg.training.print_freq == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    'Grad: {grad_norm:.4f}  '
                    'LR: {lr:.8f}  '
                    .format(epoch+1, step, len(train_loader), 
                            remain=utils.timeSince(start, float(step+1)/len(train_loader)),
                            loss=losses,
                            grad_norm=grad_norm,
                            lr=scheduler.get_lr()[0]))

            if cfg.wandb.enable:
                wandb.log({f"[fold{fold}] loss": losses.val,
                        f"[fold{fold}] lr": scheduler.get_lr()[0]})
        

        avg_loss = losses.avg
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, cfg.device)        
        score, scores = utils.get_score(valid_labels, predictions)

        elapsed = time.time() - start_time
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
        
        if cfg.wandb.enable:
            wandb.log({f"[fold{fold}] epoch": epoch+1, 
                       f"[fold{fold}] avg_train_loss": avg_loss, 
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] score": score})
        
        if best_score > score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            if hasattr(cfg.architecture, "save_name"):
                torch.save(
                    {'model': model.state_dict(),
                    'predictions': predictions},
                    f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.save_name.replace('/', '-')}/checkpoint_{fold}.pth"
                )
            else:
                torch.save(
                    {'model': model.state_dict(),
                    'predictions': predictions},
                    f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}/checkpoint_{fold}.pth"
                )




    if hasattr(cfg.architecture, "save_name"):
        predictions = torch.load(f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.save_name.replace('/', '-')}/checkpoint_{fold}.pth", 
                            map_location=torch.device('cpu'))['predictions']
    else:
        predictions = torch.load(f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}/checkpoint_{fold}.pth", 
                                map_location=torch.device('cpu'))['predictions']

    val_df[[f"pred_{c}" for c in cfg.dataset.target_cols]] = predictions
    print("*"*100)

    torch.cuda.empty_cache()
    gc.collect()
    
    return val_df



############ start here ###

def get_model(cfg):
    Net = importlib.import_module(cfg.model_class).Net
    return Net(cfg)


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs


# setting up config

parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())
for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)
print(cfg)

if hasattr(cfg.architecture, "save_name"):
    os.makedirs(f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.save_name.replace('/', '-')}", exist_ok=True)
else:
    os.makedirs(f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}", exist_ok=True)


cfg.CustomDataset = importlib.import_module(cfg.dataset_class).CustomDataset
device = "cuda"
cfg.device = device


if __name__ == "__main__":

    LOGGER = utils.get_logger(cfg)
    
    if cfg.wandb.enable: utils.init_wandb(cfg)

    if cfg.environment.seed < 0:
        cfg.environment.seed = np.random.randint(1_000_000)
    else:
        cfg.environment.seed = cfg.environment.seed

    utils.set_seed(cfg.environment.seed)

    if not os.path.exists(f"{cfg.dataset.base_dir}/train_{cfg.dataset.num_folds}folds.csv"):
        utils.create_folds(cfg)
        # utils.create_balanced_folds(cfg)

    if cfg.dataset.train_dataframe.endswith(".pq"):
        train_df = pd.read_parquet(cfg.dataset.train_dataframe)
    else:
        train_df = pd.read_csv(cfg.dataset.train_dataframe)


    if cfg.dataset.preprocess:
        train_df['full_text'] = train_df['full_text'].apply(preprocess_text)
     

    # if cfg.dataset.max_len != 1428 :
    #     tokenizer = AutoTokenizer.from_pretrained(cfg.architecture.model_name)
    #     lengths = []
    #     tk0 = tqdm(train_df['full_text'].fillna("").values, total=len(train_df))
    #     for text in tk0:
    #         length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    #         lengths.append(length)
    #     cfg.dataset.max_len = max(lengths) + 2 # cls & sep & sep
    #     print(f"max_len: {cfg.dataset.max_len}")
    

    print(f"max_len: {cfg.dataset.max_len}")
    LOGGER.info(f'CFG {cfg}')

    print(cfg.architecture.model_name)

    
    cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.architecture.model_name)
    if hasattr(cfg.architecture, "save_name"):
        cfg.tokenizer.save_pretrained(f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.save_name.replace('/', '-')}/tokenizer/")
    else:
        cfg.tokenizer.save_pretrained(f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}/tokenizer/")

    # collate_fn = DataCollatorWithPadding(tokenizer=cfg.tokenizer, max_length=cfg.dataset.max_len)


    oof_df = pd.DataFrame()
    for fold in range(cfg.dataset.num_folds):
        if cfg.dataset.fold == -1:
            fold = 0
   
        _oof_df = train_loop(train_df, fold)
        oof_df = pd.concat([oof_df, _oof_df])
        LOGGER.info(f"========== fold: {fold} result ==========")
        utils.get_result(oof_df, cfg, LOGGER)


    oof_df = oof_df.reset_index(drop=True)
    LOGGER.info(f"========== CV ==========")
    utils.get_result(oof_df, cfg, LOGGER)
    
    if hasattr(cfg.architecture, "save_name"):
        oof_df.to_csv(f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.save_name.replace('/', '-')}/oof_df.csv")
    else:
        oof_df.to_csv(f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}/oof_df.csv")

    if cfg.wandb.enable: wandb.finish()
