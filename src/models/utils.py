from .custom_model import CustomModel
import os
import torch
from transformers import AutoConfig


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def get_backbone_config(config):
    if config.model.backbone_config_path == '':
        backbone_config = AutoConfig.from_pretrained(config.model.backbone_type, output_hidden_states=True)

        backbone_config.hidden_dropout = config.model.backbone_hidden_dropout
        backbone_config.hidden_dropout_prob = config.model.backbone_hidden_dropout_prob
        backbone_config.attention_dropout = config.model.backbone_attention_dropout
        backbone_config.attention_probs_dropout_prob = config.model.backbone_attention_probs_dropout_prob

    else:
        backbone_config = torch.load(config.model.backbone_config_path)
    return backbone_config


def update_old_state(state):
    new_state = {}
    for key, value in state['model'].items():
        new_key = key
        if key.startswith('model.'):
            new_key = key.replace('model', 'backbone')
        new_state[new_key] = value

    updated_state = {'model': new_state, 'predictions': state['predictions']}
    return updated_state


def get_model(config, backbone_config_path=None, model_checkpoint_path=None, train=True):
    backbone_config = get_backbone_config(config) if backbone_config_path is None else torch.load(backbone_config_path)

    model = CustomModel(config, backbone_config=backbone_config)

    if model_checkpoint_path is not None:
        state = torch.load(model_checkpoint_path, map_location='cpu')
        if 'model.embeddings.position_ids' in state['model'].keys():
            state = update_old_state(state)
        model.load_state_dict(state['model'])

    if config.model.gradient_checkpointing:
        if model.backbone.supports_gradient_checkpointing:
            model.backbone.gradient_checkpointing_enable()
        else:
            print(f'{config.model.backbone_type} does not support gradient checkpointing')

    if train:
        if config.model.freeze_embeddings:
            freeze(model.backbone.embeddings)
        if config.model.freeze_n_layers > 0:
            freeze(model.backbone.encoder.layer[:config.model.freeze_n_layers])
        if config.model.reinitialize_n_layers > 0:
            for module in model.backbone.encoder.layer[-config.model.reinitialize_n_layers:]:
                model._init_weights(module)

    return model
