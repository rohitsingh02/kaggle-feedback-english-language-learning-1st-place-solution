import os
import gc
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import DataCollatorWithPadding
from utils import load_filepaths, str_to_bool
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


models = {
    'exp01_fb3_part2': {
            "model_name": "microsoft/deberta-v3-base",
            "model_path": "exp01-fb3-part2/microsoft-deberta-v3-base-pl",
            "tok_path": "exp01-fb3-part2/microsoft-deberta-v3-base-pl/tokenizer/",
            "pool": "MeanPool",
            "batch_size": 24,
            "model_id": "exp01_fb3_part2"
    },
    'exp01_fb3': {
            "model_name": "microsoft/deberta-v3-large",
            "model_path": "exp01-fb3/microsoft-deberta-v3-large",
            "tok_path": "exp01-fb3/microsoft-deberta-v3-large/tokenizer/",
            "pool": "MeanPool",
            "batch_size": 16,
            "model_id": "exp01_fb3"
        },
    'exp12_fb3':    {
            "model_name": "roberta-large",
            "model_path": "exp12-fb3",
            "tok_path": "exp12-fb3/tokenizer/",
            "pool": "ConcatPool",
            "batch_size": 8,
            "model_id": "exp12_fb3"
        },

    'exp14_fb3_deberta':    {
            "model_name": "microsoft/deberta-v3-base",
            "model_path": "exp14-fb3/microsoft-deberta-v3-base",
            "tok_path": "exp14-fb3/microsoft-deberta-v3-base/tokenizer/",
            "pool": "GeM",
            "batch_size": 12,
            "model_id": "exp14_fb3_deberta"
        },
    'exp14_fb3_roberta':    {
            "model_name": "roberta-large",
            "model_path": "exp14-fb3/roberta-large",
            "tok_path": "exp14-fb3/roberta-large/tokenizer/",
            "pool": "GeM",
            "batch_size": 6,
            "model_id": "exp14_fb3_roberta"
        },
    'exp02_fb3':    {
            "model_name": "microsoft/deberta-v3-large",
            "model_path": "exp02-fb3",
            "tok_path": "exp02-fb3/tokenizer/",
            "pool": "ConcatPool",
            "batch_size": 6,
            "model_id": "exp02_fb3"
        },

    'exp13_fb3':    {
            "model_name": "microsoft/deberta-v3-large",
            "model_path": "exp13-fb3",
            "tok_path": "exp13-fb3/tokenizer/",
            "pool": "WLP",
            "batch_size": 6,
            "model_id": "exp13_fb3"
        },

    'exp02_fb3_part2_distilbert':    {
            "model_name": "distilbert-base-uncased",
            "model_path": "exp02-fb3-part2/distilbert-base-uncased/distilbert-base-uncased",
            "tok_path": "exp02-fb3-part2/distilbert-base-uncased/distilbert-base-uncased/tokenizer/",
            "pool": "ConcatPool",
            "batch_size": 12,
            "model_id": "exp02_fb3_part2_distilbert"
        },

    'exp02_fb3_part2_distilbart':    {
            "model_name": "sshleifer/distilbart-cnn-12-6",
            "model_path": "exp02-fb3-part2/sshleifer-distilbart-cnn-12-6/sshleifer-distilbart-cnn-12-6",
            "tok_path": "exp02-fb3-part2/sshleifer-distilbart-cnn-12-6/sshleifer-distilbart-cnn-12-6/tokenizer/",
            "pool": "ConcatPool",
            "batch_size": 6,
            "model_id": "exp02_fb3_part2_distilbart"
        },

    'exp03_fb3':    {
            "model_name": "roberta-large",
            "model_path": "exp03-fb3/roberta-large",
            "tok_path": "exp03-fb3/roberta-large/tokenizer/",
            "pool": "WLP",
            "batch_size": 6,
            "model_id": "exp03_fb3"
        },
    'exp13_part2':    {
            "model_name": "microsoft/deberta-v2-xlarge",
            "model_path": "exp13-part2/microsoft-deberta-v2-xlarge",
            "tok_path": "exp13-part2/microsoft-deberta-v2-xlarge/tokenizer/",
            "pool": "WLP",
            "batch_size": 6,
            "model_id": "exp13_part2"
        },
    'exp04_fb3':    {
            "model_name": "microsoft/deberta-v3-base",
            "model_path": "exp04-fb3/microsoft-deberta-v3-base",
            "tok_path": "exp04-fb3/microsoft-deberta-v3-base/tokenizer/",
            "pool": "GeM",
            "batch_size": 16,
            'id': 'exp04_fb3',
            "model_id": "exp04_fb3"
        },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--debug', type=str_to_bool, default=False)
    arguments = parser.parse_args()
    return arguments


def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores


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
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)

        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


def prepare_input(text, tokenizer, model_type):
    if "roberta" in model_type or "distilbert" in model_type or "facebook/bart" in model_type or "distilbart" in model_type:
        inputs = tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            truncation=True
        )
    else:
        inputs = tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
        )

    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, df, tokenizer, model_type=None):
        self.texts = df['full_text'].values
        self.model_type = model_type
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.tokenizer, self.model_type)
        return inputs


# MeanPool
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# WLP
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, features):
        ft_all_layers = features['all_layer_embeddings']

        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        features.update({'token_embeddings': weighted_average})
        return features


# GeM
class GeMText(nn.Module):
    def __init__(self, dim=1, cfg=None, p=3, eps=1e-6):
        super(GeMText, self).__init__()
        self.dim = dim
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.feat_mult = 1
        # x seeems last hidden state

    def forward(self, x, attention_mask):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)
        x = (x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(cfg["model_name"], config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if cfg["pool"] == "MeanPool" or cfg["pool"] == "ConcatPool":
            self.pooling = MeanPooling()
        elif cfg["pool"] == "WLP":
            self.pooling = WeightedLayerPooling(self.config.num_hidden_layers, layer_start=9)
        elif cfg["pool"] == "GeM":
            self.pooling = GeMText()

        if cfg["pool"] == "ConcatPool":
            self.head = nn.Linear(self.config.hidden_size * 4, 6)
        else:
            self.head = nn.Linear(self.config.hidden_size, 6)

        if 'facebook/bart' in cfg["model_name"] or 'distilbart' in cfg["model_name"]:
            self.config.use_cache = False
            self.initializer_range = self.config.init_std
        else:
            self.initializer_range = self.config.initializer_range

        self._init_weights(self.head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        attention_mask = inputs["attention_mask"]
        input_ids = inputs["input_ids"]

        if self.cfg["pool"] == "WLP":
            x = self.model(input_ids=input_ids, attention_mask=attention_mask)
            tmp = {
                'all_layer_embeddings': x.hidden_states
            }
            feature = self.pooling(tmp)['token_embeddings'][:, 0]

        elif self.cfg["pool"] == "ConcatPool":

            if 'facebook/bart' in self.cfg["model_name"] or 'distilbart' in self.cfg["model_name"]:
                x = torch.stack(self.model(input_ids=input_ids, attention_mask=attention_mask,
                                           output_hidden_states=True).decoder_hidden_states)
            else:
                x = torch.stack(self.model(input_ids=input_ids, attention_mask=attention_mask).hidden_states)

            p1 = self.pooling(x[-1], attention_mask)
            p2 = self.pooling(x[-2], attention_mask)
            p3 = self.pooling(x[-3], attention_mask)
            p4 = self.pooling(x[-4], attention_mask)

            feature = torch.cat(
                (p1, p2, p3, p4), -1
            )
        else:
            outputs = self.model(**inputs)
            x = outputs[0]
            feature = self.pooling(x, inputs['attention_mask'])

        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.head(feature)
        return output


if __name__ == '__main__':
    seed_everything(seed=42)
    args = parse_args()

    filepaths = load_filepaths()
    assert args.model_id in list(models.keys()), f'Model id should be one of: {list(models.keys())}'

    cfg = models[args.model_id]
    cfg['model_path'] = os.path.join(filepaths['MODELS_DIR_PATH'], cfg['model_path'])
    cfg['tok_path'] = os.path.join(filepaths['MODELS_DIR_PATH'], cfg['tok_path'])

    num_workers = 4
    gradient_checkpointing = False
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    tokenizer = AutoTokenizer.from_pretrained(f"{cfg['tok_path']}")

    test = pd.read_csv(filepaths['TRAIN_CSV_PATH'])
    submission = pd.read_csv(filepaths['SAMPLE_SUBMISSION_CSV_PATH'])

    test['tokenize_length'] = [len(tokenizer(text)['input_ids']) for text in test['full_text'].values]
    test = test.sort_values('tokenize_length', ascending=True).reset_index(drop=True)

    if args.debug:
        test_df = test.sample(50, random_state=1)

    all_preds = []

    tokenizer = AutoTokenizer.from_pretrained(f"{cfg['tok_path']}")
    test_dataset = TestDataset(test, tokenizer, cfg["model_name"])
    batch_size = cfg["batch_size"]

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='longest'),
                             num_workers=2, pin_memory=True, drop_last=False)

    predictions = []
    print(cfg['model_path'])

    for fold in trn_fold:

        model = CustomModel(cfg, config_path=cfg["model_path"] + "/config.pth", pretrained=False)
        state = torch.load(f"{cfg['model_path']}/checkpoint_{fold}.pth",
                           map_location=torch.device('cpu'))

        model.load_state_dict(state['model'])
        prediction = inference_fn(test_loader, model, device)
        predictions.append(prediction)

        out = test.copy()
        out[target_cols] = prediction

        pseudo_path = filepaths['PREVIOUS_DATA_PSEUDOLABELS_DIR_PATH']

        dir_path = os.path.join(pseudo_path, f'{cfg["model_id"]}_pseudolabels')
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        out.to_csv(os.path.join(dir_path, f'pseudolabels_fold{fold}.csv'), index=False)

        del model, state, prediction
        gc.collect()
        torch.cuda.empty_cache()

    predictions = np.mean(predictions, axis=0)
    all_preds.append(predictions)
    del tokenizer, test_dataset, test_loader
    gc.collect()



