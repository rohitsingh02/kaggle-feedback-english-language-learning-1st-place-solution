from transformers import get_linear_schedule_with_warmup, \
  get_cosine_schedule_with_warmup, \
  get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup


def get_scheduler(optimizer, config, num_train_steps):

    if config.scheduler.scheduler_type == 'constant_schedule_with_warmup':
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.constant_schedule_with_warmup.n_warmup_steps
        )
    elif config.scheduler.scheduler_type == 'linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.linear_schedule_with_warmup.n_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif config.scheduler.scheduler_type == 'cosine_schedule_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.cosine_schedule_with_warmup.n_warmup_steps,
            num_cycles=config.scheduler.cosine_schedule_with_warmup.n_cycles,
            num_training_steps=num_train_steps,
        )
    elif config.scheduler.scheduler_type == 'polynomial_decay_schedule_with_warmup':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.polynomial_decay_schedule_with_warmup.n_warmup_steps,
            num_training_steps=num_train_steps,
            power=config.scheduler.polynomial_decay_schedule_with_warmup.power,
            lr_end=config.scheduler.polynomial_decay_schedule_with_warmup.min_lr
        )
    else:
        raise ValueError(f'Unknown scheduler: {config.scheduler.scheduler_type}')

    return scheduler

