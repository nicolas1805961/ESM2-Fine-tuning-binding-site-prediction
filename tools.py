import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from ruamel.yaml import YAML
import json
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

def get_data():
    # Open and read the JSON file
    with open('uniprotkb_ligand_AND_reviewed_true_AND_2024_10_02.json', 'r') as file:
        data = json.load(file)['results']

    # Print the data
    out = {'entry': [], 'sequence': [], 'length': [], 'binding_site': [], 'label': []}
    for i in range(len(data)):
        prot = data[i]
        out['entry'].append(prot['primaryAccession'])
        out['sequence'].append(prot['sequence']['value'])
        out['length'].append(prot['sequence']['length'])
        binding_site_list = []
        for j in range(len(prot['features'])):
            if prot['features'][j]['type'] == 'Binding site':
                start = prot['features'][j]['location']['start']['value']
                end = prot['features'][j]['location']['end']['value']
                binding_site_list+= np.arange(start, end+1).tolist()
        out['binding_site'].append(binding_site_list)

        label = np.zeros(shape=(len(prot['sequence']['value']),), dtype=int)
        label[np.array(binding_site_list).astype(int) - 1] = 1
        out['label'].append(label)

    df = pd.DataFrame(out)
    print(len(df))
    df = df[df['length'] < 1023]
    print(len(df))

    train_sequences, test_sequences, train_labels, test_labels = train_test_split(df['sequence'].values.tolist(), df['label'].values.tolist(), test_size=0.20, shuffle=True, random_state=42)
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(train_sequences, train_labels, test_size=0.20, shuffle=True, random_state=42)
    
    return train_sequences, test_sequences, val_sequences, train_labels, test_labels, val_labels


def read_config_video(filename):
    yaml = YAML()
    with open(filename) as file:
        config = yaml.load(file)

    return config


def load_data():
    with open('train.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        train_sequences = loaded_dict['X']
        train_labels = loaded_dict['Y']

    with open('test.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        test_sequences = loaded_dict['X']
        test_labels = loaded_dict['Y']

    with open('val.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        val_sequences = loaded_dict['X']
        val_labels = loaded_dict['Y']

    return train_sequences, test_sequences, val_sequences, train_labels, test_labels, val_labels