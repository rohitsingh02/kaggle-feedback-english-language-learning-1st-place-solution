import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding


class FeedbackDataset(Dataset):
    def __init__(self, cfg, df, train=True):
        self.cfg = cfg
        self.df = df
        self.texts = self.df['full_text'].values

        self.labels = None
        if cfg.general.target_columns[0] in df.columns and train:
            self.labels = df[cfg.general.target_columns].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        inputs = self.cfg.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.cfg.general.max_length,
            pad_to_max_length=True,
            truncation=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[item], dtype=torch.float)
            return inputs, label
        return inputs


def get_train_dataloader(cfg, df):
    dataset = FeedbackDataset(cfg, df)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.general.train_batch_size,
        num_workers=cfg.general.n_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def get_valid_dataloader(cfg, df):
    dataset = FeedbackDataset(cfg, df)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.general.valid_batch_size,
        num_workers=cfg.general.n_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader


def get_test_dataloader(cfg, df):
    dataset = FeedbackDataset(cfg, df, train=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.general.valid_batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=cfg.tokenizer, padding='longest'),
        num_workers=cfg.general.n_workers,
        pin_memory=True,
        drop_last=False
    )
    return dataloader
