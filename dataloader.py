import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
import monai.transforms as T
from torchvision import transforms
import pickle
from datasets import Dataset

def collate_fn(batch):
    print(len(batch))
    print(len(batch[0]))
    out = {}
    out['input_ids'] = torch.stack([batch[i][0] for i in range(len(batch))])
    out['attention_mask'] = torch.stack([batch[i][1] for i in range(len(batch))])
    out['labels'] = torch.stack([batch[i][2] for i in range(len(batch))])

    return out

class CustomDataloaderTrain(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x, y, tokenizer):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        x = tokenizer(x)
        self.X = Dataset.from_dict(x)
        self.X = self.X.add_column("labels", y)

        #print(tokenizer)
        #for i in range(len(self.X['input_ids'])):
        #    special_token_mask = tokenizer.get_special_tokens_mask(self.X['input_ids'][i])
        #    nb_special_tokens = np.count_nonzero(special_token_mask)
        #    assert nb_special_tokens == 2
        #    assert special_token_mask[0] == special_token_mask[-1] == 1
        #    y[i] = np.insert(y[i], 0, -100, axis=0)
        #    y[i] = np.append(y[i], -100)
        #    assert len(y[i]) == len(self.X['input_ids'][i])
        #self.Y = y

        #self.path_list = self.path_list[:10]


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = {key: torch.tensor(value[idx]) for key, value in self.X.items()}
        
        Y = self.Y[idx]

        X.update({'labels': torch.from_numpy(Y)})

        #print(X)
        #print(len(X['input_ids']))
        #print(len(X['attention_mask']))
        #print(len(X['labels']))
        #print(type(X))

        return X
    



class CustomDataloaderVal(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x, y, tokenizer):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.X = tokenizer(x)
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X = self.X[idx]
        Y = self.Y[idx]

        return X.update({'labels': Y})