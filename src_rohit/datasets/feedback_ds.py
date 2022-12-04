# from cfg import CFG, CFG1
# from torch.utils.data import Dataset, DataLoader


# class CustomDataset(Dataset):
#     def __init__(self, df, tokenizer, max_length):
#         self.df = df
#         self.max_len = max_length
#         self.tokenizer = tokenizer
#         self.texts = df['full_text'].values
#         self.targets = df[CFG.target_cols].values
        
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, index):
#         text = self.texts[index]
#         inputs = self.tokenizer.encode_plus(
#                         text,
#                         truncation=True,
#                         add_special_tokens=True,
#                         max_length=self.max_len
#                     )
        
#         return {
#             'input_ids': inputs['input_ids'],
#             'attention_mask': inputs['attention_mask'],
#             'target': self.targets[index]
#         }


from torch.utils.data import Dataset, DataLoader
import torch
import collections
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self, df, mode, cfg):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.cfg = cfg

        self.tokenizer = cfg.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        self.texts = self.df[cfg.dataset.text_column].values
        self.targets = self.df[cfg.dataset.target_cols].values
        self.cfg._tokenizer_cls_token_id = self.tokenizer.cls_token_id
        self.cfg._tokenizer_sep_token_id = self.tokenizer.sep_token_id
        self.cfg._tokenizer_mask_token_id = self.tokenizer.mask_token_id


    def __len__(self):
        return len(self.df)


    def batch_to_device(batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, collections.abc.Mapping):
            return {
                key: CustomDataset.batch_to_device(value, device)
                for key, value in batch.items()
            }
        elif isinstance(batch, collections.abc.Sequence):
            return [CustomDataset.batch_to_device(value, device) for value in batch]
        else:
            raise ValueError(f"Can not move {type(batch)} to device.")


    def encode(self, text):
        sample = dict()
        encodings = self.tokenizer.encode_plus(
            text, 
            return_tensors=None, 
            add_special_tokens=True, 
            max_length=self.cfg.dataset.max_len,
            pad_to_max_length=True,
            truncation=True
        )

        sample["input_ids"] =  torch.tensor(encodings["input_ids"], dtype=torch.long) 
        sample["attention_mask"] = torch.tensor(encodings["attention_mask"], dtype=torch.long) 
        return sample


    def _read_data(self, idx, sample):
        text = self.texts[idx][0]
        # if idx == 0:
        #     print(text)

        # sample = self.encode(text, sample)
        sample.update(self.encode(text))
        return sample

    def _read_label(self, idx, sample):
    
        sample["target"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return sample


    def __getitem__(self, idx):
        sample = dict()
        sample = self._read_data(idx=idx, sample=sample)
        if self.targets is not None:
            sample = self._read_label(idx=idx, sample=sample)

        return sample