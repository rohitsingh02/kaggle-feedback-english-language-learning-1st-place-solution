import os
import torch
import random
import numpy as np
import pandas as pd
import math 
import time
from torch.optim import Optimizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import mean_squared_error


import codecs
import os
import gc
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, AdamW, DataCollatorWithPadding
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from text_unidecode import unidecode



def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_folds(cfg):
    df = pd.read_csv(cfg.dataset.comp_train_dataframe)
    mskf = MultilabelStratifiedKFold(n_splits=cfg.dataset.num_folds, shuffle=True, random_state=cfg.environment.seed)

    for fold, ( _, val_) in enumerate(mskf.split(X=df, y=df[cfg.dataset.target_cols])):
        df.loc[val_ , "fold"] = int(fold)
        
    df["fold"] = df["fold"].astype(int)
    df.to_csv(f"{cfg.dataset.base_dir}/train_{cfg.dataset.num_folds}folds.csv", index=False)



def get_text_token_cnts(
    df: pd.DataFrame,
    tokenizer,
    inplace: bool = False
):
    token_cnts = []
    for text in tqdm(df["full_text"], total=len(df)):
        input_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False
        )["input_ids"]
        token_cnts.append([len(input_ids), len(set(input_ids))])
    if not inplace:
        df = df.copy()
    df[["n_token", "nunique_token"]] = token_cnts
    return df


def create_balanced_folds(cfg):
    df = pd.read_csv(cfg.dataset.comp_train_dataframe) 
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    df = get_text_token_cnts(df, tokenizer)
    
    N_TOKEN_BIN = 50
    N_UNIQUE_TOKEN_BIN = 50
    df["n_token_bin"] = pd.qcut(
        x=df["n_token"], q=N_TOKEN_BIN, labels=False, duplicates="drop"
    )
    df["nutoken_bin"] = pd.qcut(
        x=df["nunique_token"], q=N_UNIQUE_TOKEN_BIN, labels=False, duplicates="drop"
    )
    
    
    mskf = MultilabelStratifiedKFold(n_splits=cfg.dataset.num_folds, shuffle=True, random_state=cfg.environment.seed)

    for fold, ( _, val_) in enumerate(mskf.split(X=df, y=df[cfg.dataset.target_cols + ["n_token_bin", "nutoken_bin"] ])):
        df.loc[val_ , "fold"] = int(fold)
        
    df["fold"] = df["fold"].astype(int)
    df.to_csv(f"{cfg.dataset.base_dir}/train_balanced_{cfg.dataset.num_folds}folds.csv", index=False)





def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores


def get_result(oof_df, cfg, logger):
    labels = oof_df[cfg.dataset.target_cols].values
    preds = oof_df[[f"pred_{c}" for c in cfg.dataset.target_cols]].values
    score, scores = get_score(labels, preds)
    logger.info(f'Score: {score:<.4f}  Scores: {scores}')




# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def get_logger(cfg):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    
    if hasattr(cfg.architecture, "save_name"):
        filename=f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.save_name.replace('/', '-')}/train"
    else:
        filename=f"{cfg.output_dir}/{cfg.experiment_name}/{cfg.architecture.model_name.replace('/', '-')}/train"

    
    
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    filename=filename
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger



### setup wand


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

def init_wandb(cfg):
    import wandb
    try:
        wandb.login(key="39a298fe785a51ae22d755b11a9f9fff01321796")
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')




    run = wandb.init(project=cfg.wandb.project_name, 
                     name=f"{cfg.experiment_name}-{cfg.architecture.model_name}",
                     config=class2dict(cfg),
                     group=cfg.architecture.model_name,
                     job_type="train",
                     anonymous=anony)

    return run


class PriorWD(Optimizer):
    def __init__(self, optim, use_prior_wd=False, exclude_last_group=True):
        super(PriorWD, self).__init__(optim.param_groups, optim.defaults)
        self.param_groups = optim.param_groups
        self.optim = optim
        self.use_prior_wd = use_prior_wd
        self.exclude_last_group = exclude_last_group
        self.weight_decay_by_group = []
        for i, group in enumerate(self.param_groups):
            self.weight_decay_by_group.append(group["weight_decay"])
            group["weight_decay"] = 0

        self.prior_params = {}
        for i, group in enumerate(self.param_groups):
            for p in group["params"]:
                self.prior_params[id(p)] = p.detach().clone()

    def step(self, closure=None):
        if self.use_prior_wd:
            for i, group in enumerate(self.param_groups):
                for p in group["params"]:
                    if self.exclude_last_group and i == len(self.param_groups):
                        p.data.add_(-group["lr"] * self.weight_decay_by_group[i], p.data)
                    else:
                        p.data.add_(
                            -group["lr"] * self.weight_decay_by_group[i], p.data - self.prior_params[id(p)],
                        )
        loss = self.optim.step(closure)

        return loss

    def compute_distance_to_prior(self, param):
        assert id(param) in self.prior_params, "parameter not in PriorWD optimizer"
        return (param.data - self.prior_params[id(param)]).pow(2).sum().sqrt()





import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd.function import InplaceFunction

class Mixout(InplaceFunction):
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, training=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("A mix probability of mixout has to be between 0 and 1," " but got {}".format(p))
        if target is not None and input.size() != target.size():
            raise ValueError(
                "A target tensor size must match with a input tensor size {},"
                " but got {}".format(input.size(), target.size())
            )
        ctx.p = p
        ctx.training = training

        if ctx.p == 0 or not ctx.training:
            return input

        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        ctx.noise.expand_as(input)

        if ctx.p == 1:
            output = target
        else:
            output = ((1 - ctx.noise) * target + ctx.noise * output - ctx.p * target) / (1 - ctx.p)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None


def mixout(input, target=None, p=0.0, training=False, inplace=False):
    return Mixout.apply(input, target, p, training, inplace)


class MixLinear(torch.nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]
    def __init__(self, in_features, out_features, bias=True, target=None, p=0.0):
        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.target = target
        self.p = p

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, mixout(self.weight, self.target, self.p, self.training), self.bias)

    def extra_repr(self):
        type = "drop" if self.target is None else "mix"
        return "{}={}, in_features={}, out_features={}, bias={}".format(
            type + "out", self.p, self.in_features, self.out_features, self.bias is not None
        )



### preprocessing start here

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text