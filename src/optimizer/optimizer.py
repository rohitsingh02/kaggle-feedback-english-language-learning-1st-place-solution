from .parameters import get_grouped_llrd_parameters, get_optimizer_params
from torch.optim import AdamW
from torchcontrib.optim import SWA


def get_optimizer(model, config):
    
    if config.optimizer.group_lt_multiplier == 1:
        optimizer_parameters = get_optimizer_params(model,
                                                    config.optimizer.encoder_lr,
                                                    config.optimizer.decoder_lr,
                                                    weight_decay=config.optimizer.weight_decay)
    else:
        optimizer_parameters = get_grouped_llrd_parameters(model,
                                                           encoder_lr=config.optimizer.encoder_lr,
                                                           decoder_lr=config.optimizer.decoder_lr,
                                                           embeddings_lr=config.optimizer.embeddings_lr,
                                                           lr_mult_factor=config.optimizer.group_lt_multiplier,
                                                           weight_decay=config.optimizer.weight_decay,
                                                           n_groups=config.optimizer.n_groups)

    optimizer = AdamW(optimizer_parameters,
                      lr=config.optimizer.encoder_lr,
                      eps=config.optimizer.eps,
                      betas=config.optimizer.betas)

    if config.optimizer.use_swa:
        optimizer = SWA(optimizer,
                        swa_start=config.optimizer.swa.swa_start,
                        swa_freq=config.optimizer.swa.swa_freq,
                        swa_lr=config.optimizer.swa.swa_lr)
    return optimizer
