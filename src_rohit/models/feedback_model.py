import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch.nn.parameter import Parameter


import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch.nn.parameter import Parameter
from transformers import AutoTokenizer


import utils


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class WeightedDenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)

        if weights is not None:
            loss = loss * weights
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.mean()

        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor, weights=None):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor.argmax(dim=1),
            weight=self.weight,
            reduction=self.reduction,
        )




import torch.nn as nn
import torch
from sklearn.metrics import mean_squared_error
import numpy as np

class RMSELoss(nn.Module):
    """
    Code taken from Y Nakama's notebook (https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
    """
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, predictions, targets):
        loss = torch.sqrt(self.mse(predictions, targets) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


##################################################################################

################################    Poolings    ##################################

##################################################################################

class MeanPooling(nn.Module):
    def __init__(self, dim, cfg):
        super(MeanPooling, self).__init__()
        self.feat_mult = 1
        
    def forward(self, x, attention_mask, input_ids, cfg):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings



class MeanMaxPooling(nn.Module):
    def __init__(self, dim, cfg):
        super(MeanMaxPooling, self).__init__()
        self.feat_mult = 1
        
    def forward(self, x, attention_mask, input_ids, cfg):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask 

        embeddings = x.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        mean_max_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)

        return mean_max_embeddings
    


class MaxPooling(nn.Module):
    def __init__(self, dim, cfg):
        super(MaxPooling, self).__init__()
        self.feat_mult = 1
        
    def forward(self, x, attention_mask, input_ids, cfg):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        embeddings = x.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        return max_embeddings
    


class MinPooling(nn.Module):
    def __init__(self, dim, cfg,):
        super(MinPooling, self).__init__()
        self.feat_mult = 1
        
    def forward(self, x, attention_mask, input_ids, cfg):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        embeddings = x.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim = 1)
        return min_embeddings



class GeMText(nn.Module):
    def __init__(self, dim=1, cfg=None, p=3, eps=1e-6):
        super(GeMText, self).__init__()
        self.dim = dim
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.feat_mult = 1
        # x seeems last hidden state

    def forward(self, x, attention_mask, input_ids, cfg):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)
        x = (x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, features):
        ft_all_layers = features['all_layer_embeddings']

        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        features.update({'token_embeddings': weighted_average})
        return features



class AttentionPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_fc):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_fc = hiddendim_fc
        self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float()
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()



    def forward(self, all_hidden_states):
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q.to("cuda"), h.transpose(-2, -1).to("cuda")).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0).to("cuda"), v_temp.to("cuda")).squeeze(2)
        return v



class NLPPoolings:
    _poolings = {
        # "All [CLS] token": NLPAllclsTokenPooling,
        "GeM": GeMText,
        "Mean": MeanPooling,
        "Max": MaxPooling,
        "Min": MinPooling,
        "MeanMax": MeanMaxPooling,
        "WLP": WeightedLayerPooling,
        "ConcatPool":MeanPooling,
        "AP": AttentionPooling
    }

    @classmethod
    def get(cls, name):
        return cls._poolings.get(name)



class Net(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=True):
        super(Net, self).__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.architecture.model_name, output_hidden_states=True)
            # self.config.update({'max_position_embeddings':768})             
            print(self.config)


            if 'roberta-base' in cfg.architecture.model_name:
                print()

            if cfg.architecture.custom_intermediate_dropout:
                self.config.hidden_dropout = cfg.architecture.intermediate_dropout
                self.config.hidden_dropout_prob = cfg.architecture.intermediate_dropout
                self.config.attention_dropout = cfg.architecture.intermediate_dropout
                self.config.attention_probs_dropout_prob = cfg.architecture.intermediate_dropout
            # if logger: logger.info(self.config)
        else:
            self.config = torch.load(config_path)

        self.model = AutoModel.from_pretrained(cfg.architecture.model_name, config=self.config)
        # self.model = SwitchTransformersConditionalGeneration.from_pretrained(cfg.architecture.model_name, device_map="auto")


        if hasattr(self.cfg.architecture, "mixout") and cfg.architecture.mixout > 0:
            print('Initializing Mixout Regularization')
            for sup_module in self.model.modules():
                for name, module in sup_module.named_children():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.0
                    if isinstance(module, nn.Linear):
                        target_state_dict = module.state_dict()
                        bias = True if module.bias is not None else False
                        new_module = utils.MixLinear(
                            module.in_features, module.out_features, bias, target_state_dict["weight"], cfg.architecture.mixout
                        )
                        new_module.load_state_dict(target_state_dict)
                        setattr(sup_module, name, new_module)
            print('Done.!')


        if self.cfg.architecture.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.pooling = NLPPoolings.get(self.cfg.architecture.pool)
        if self.cfg.architecture.pool == "WLP":
            self.pooling = self.pooling(self.config.num_hidden_layers, layer_start=cfg.training.layer_start)
        elif self.cfg.architecture.pool == "AP":
            hiddendim_fc = 128
            self.pooling = AttentionPooling(self.config.num_hidden_layers, self.config.hidden_size, hiddendim_fc)
        else:
            self.pooling = self.pooling(dim=1, cfg=cfg)

        if self.cfg.architecture.pool == "MeanMax":
            self.head = nn.Linear(self.config.hidden_size*2, len(cfg.dataset.target_cols))
        elif self.cfg.architecture.pool == "ConcatPool":
            if 'distilbert-base-uncased' in self.cfg.architecture.model_name:
                self.head = nn.Linear(self.config.hidden_size*4, len(cfg.dataset.target_cols))
            else:
                self.head = nn.Linear(self.config.hidden_size*4, len(cfg.dataset.target_cols))
        elif self.cfg.architecture.pool == "AP":
            self.head = nn.Linear(hiddendim_fc, len(cfg.dataset.target_cols)) # regression head
        else:
            self.head = nn.Linear(self.config.hidden_size, len(cfg.dataset.target_cols))
        
        # print(self.config)
        
        
        if 'facebook/bart' in cfg.architecture.model_name or 'distilbart' in cfg.architecture.model_name:
            self.config.use_cache = False
            self.initializer_range = self.config.init_std
        else:
            self.initializer_range = self.config.initializer_range
        
        
        print(self.model)
        self._init_weights(self.head)

        if hasattr(self.cfg.architecture, "reinit_n_layers") and cfg.architecture.reinit_n_layers > 0:
            self.reinit_n_layers = cfg.architecture.reinit_n_layers
            self._do_reinit()    


        if 'deberta-v3-base' in cfg.architecture.model_name or 'mdeberta-v3-base' in cfg.architecture.model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:9].requires_grad_(False) # exp15 45009
            
        elif 'deberta-v3-large' in cfg.architecture.model_name or 'deberta-v2-xlarge' in cfg.architecture.model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:18].requires_grad_(False) # exp15 45009

        elif 'deberta-v3-small' in cfg.architecture.model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:1].requires_grad_(False) # 冻结1/6

        # elif 'distilbert-base-uncased' in cfg.architecture.model_name:
        #     self.model.embeddings.requires_grad_(False)
        #     self.model.transformer.layer[:2].requires_grad_(False) # 冻结1/6
            

        if self.cfg.training.loss_function == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.cfg.training.loss_function == 'rmse':
            self.loss_fn = RMSELoss(reduction="mean")
        elif self.cfg.training.loss_function == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction="mean")
            self.loss_fn2 = nn.HuberLoss(reduction="mean", delta=0.5) # 0.5
        elif self.cfg.training.loss_function == "CrossEntropy":
            self.loss_fn = DenseCrossEntropy()
        elif self.cfg.training.loss_function == "WeightedCrossEntropy":
            self.loss_fn = WeightedDenseCrossEntropy()
        elif self.cfg.training.loss_function == "FocalLoss":
            self.loss_fn = FocalLoss()



    def _do_reinit(self):
        # Re-init last n layers.
        for n in range(self.reinit_n_layers):
            # print(self.model)
            if 'facebook/bart' in self.cfg.architecture.model_name or 'distilbart' in self.cfg.architecture.model_name:
                self.model.encoder.layers[-(n+1)].apply(self._init_weights)
            elif 'funnel' in self.cfg.architecture.model_name:
                self.model.decoder.layers[-(n+1)].apply(self._init_weights)
            elif "distilbert" in self.cfg.architecture.model_name:
                self.model.transformer.layer[-(n+1)].apply(self._init_weights)
            else:
                self.model.encoder.layer[-(n+1)].apply(self._init_weights)


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

        if (
            self.training
            and hasattr(self.cfg.training, "mask_probability")
            and self.cfg.training.mask_probability > 0 and self.cfg.epoch < 2
        ):
            input_ids = inputs["input_ids"].clone()
            special_mask = torch.ones_like(input_ids)
            special_mask[
                (input_ids == self.cfg._tokenizer_cls_token_id)
                | (input_ids == self.cfg._tokenizer_sep_token_id)
                | (input_ids >= self.cfg._tokenizer_mask_token_id)
            ] = 0

            mask = (
                torch.bernoulli(
                    torch.full(input_ids.shape, self.cfg.training.mask_probability)
                )
                .to(input_ids.device)
                .bool()
                & special_mask.bool()
            ).bool()
            input_ids[mask] = self.cfg._tokenizer_mask_token_id
            inputs["input_ids"] = input_ids.clone()

        # x = self.model(
        #     input_ids=input_ids, attention_mask=attention_mask
        # ).last_hidden_state

        if self.cfg.architecture.pool == "WLP":
            if 'facebook/bart' in self.cfg.architecture.model_name or 'distilbart' in self.cfg.architecture.model_name:
                x = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                tmp = {
                    'all_layer_embeddings': x.encoder_hidden_states
                }
            else:
                x = self.model(input_ids=input_ids, attention_mask=attention_mask)
                tmp = {
                    'all_layer_embeddings': x.hidden_states
                }
            x = self.pooling(tmp)['token_embeddings'][:, 0]

        elif self.cfg.architecture.pool == "ConcatPool":

            if 'facebook/bart' in self.cfg.architecture.model_name or 'distilbart' in self.cfg.architecture.model_name:
                x = torch.stack(self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).decoder_hidden_states)
            else:
                x = torch.stack(self.model(input_ids=input_ids, attention_mask=attention_mask).hidden_states)

            p1 = self.pooling(x[-1], attention_mask, input_ids, self.cfg)
            p2 = self.pooling(x[-2], attention_mask, input_ids, self.cfg)
            p3 = self.pooling(x[-3], attention_mask, input_ids, self.cfg)
            p4 = self.pooling(x[-4], attention_mask, input_ids, self.cfg)

            x = torch.cat(
                (p1, p2, p3, p4),-1
            )
            
            # if 'distilbert-base-uncased' in self.cfg.architecture.model_name:
            #     x = torch.cat(
            #         (p1, p2),-1
            #     )

            # x = torch.cat(
            #     (x[-1], x[-2], x[-3], x[-4]),-1
            # )

            # x = self.pooling(x, attention_mask, input_ids, self.cfg) # tmp/wrong


        elif self.cfg.architecture.pool == "AP":
            x = torch.stack(self.model(input_ids=input_ids, attention_mask=attention_mask).hidden_states)
            x = self.pooling(x)
        else:
            
            if 'facebook/bart' in self.cfg.architecture.model_name or 'distilbart' in self.cfg.architecture.model_name:
                x = self.model( input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            else:
                x = self.model( input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            x = self.pooling(x, attention_mask, input_ids, self.cfg)

        return x, inputs

    # def feature(self, inputs):
    #     outputs = self.model(**inputs)
    #     last_hidden_states = outputs[0]
    #     feature = self.pooling(last_hidden_states, inputs['attention_mask'])
    #     return feature

    def forward(self, inputs, calculate_loss=True):
        x, input = self.feature(inputs)

        if (
            hasattr(self.cfg.architecture, "wide_dropout")
            and self.cfg.architecture.wide_dropout > 0.0
            and self.training
        ):
            x1 = self.head(
                F.dropout(
                    x, p=self.cfg.architecture.wide_dropout, training=self.training
                )
            )
            x2 = self.head(
                F.dropout(
                    x, p=self.cfg.architecture.wide_dropout, training=self.training
                )
            )
            x3 = self.head(
                F.dropout(
                    x, p=self.cfg.architecture.wide_dropout, training=self.training
                )
            )
            x4 = self.head(
                F.dropout(
                    x, p=self.cfg.architecture.wide_dropout, training=self.training
                )
            )
            x5 = self.head(
                F.dropout(
                    x, p=self.cfg.architecture.wide_dropout, training=self.training
                )
            )
            logits = (x1 + x2 + x3 + x4 + x5) / 5
        else:
            logits = self.head(x)


        outputs = {}
        outputs["logits"] = logits

        if "target" in input:
            outputs["target"] = input["target"]

        if calculate_loss:
            targets = input["target"]

            outputs["loss"] =  self.loss_fn(logits, targets) 
            # outputs["loss"] = 0.7* self.loss_fn(logits, targets) + 0.3*self.loss_fn(logits, targets)

        return outputs




