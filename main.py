import torchvision
from datasets import load_dataset
import torch
from get_model import get_model
from train import train
import os
from datetime import datetime
from copy import copy
from torch.utils.tensorboard import SummaryWriter
from tools import read_config_video, load_data
import argparse
from pathlib import Path
import numpy as np
import random
import torch.backends.cudnn as cudnn
from ruamel.yaml import YAML
from transformers import DataCollatorForTokenClassification
from datasets import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("-config", help="yaml config file", required=True)
parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
args = parser.parse_args()
deterministic = args.deterministic

if deterministic:
    random.seed(12345)
    np.random.seed(12345)
    torch.cuda.manual_seed_all(12345)
    torch.manual_seed(12345)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


output_folder = 'out'

train_sequences, test_sequences, val_sequences, train_labels, test_labels, val_labels = load_data()

config = read_config_video(os.path.join(Path.cwd(), args.config))

# You can lower your batch size if you're running out of GPU memory
batch_size = config['batch_size']
epochs = config['epochs']
validation_step = config['validation_step']
model_checkpoint = config['model_checkpoint']
learning_rate = config['learning_rate']
eta_min = config['eta_min']
weight_decay = config['weight_decay']

tokenizer, model = get_model(model_checkpoint)

train_sequences = tokenizer(train_sequences)
train_sequences = Dataset.from_dict(train_sequences)
train_dataset = train_sequences.add_column("labels", train_labels)

val_sequences = tokenizer(val_sequences)
val_sequences = Dataset.from_dict(val_sequences)
val_dataset = val_sequences.add_column("labels", val_labels)

# Or load images from a local folder
#train_dataset = CustomDataloaderTrain(x=train_sequences, y=train_labels, tokenizer=tokenizer)
#val_dataset = CustomDataloaderVal(x=val_sequences, y=val_labels, tokenizer=tokenizer)

data_collator = DataCollatorForTokenClassification(tokenizer)

# Create a dataloader from the dataset to serve up the transformed images in batches
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=data_collator)

timestr = datetime.now().strftime("%Y-%m-%d_%HH%M_%Ss_%f")
log_dir = os.path.join(copy(output_folder), timestr)
writer = SummaryWriter(log_dir=log_dir)

yaml = YAML()
with open(os.path.join(log_dir, 'config.yaml'), 'wb') as f:
    yaml.dump(config, f)

train(model=model, 
      train_dataloader=train_dataloader, 
      val_dataloader=val_dataloader, 
      writer=writer, 
      epochs=epochs, 
      validation_step=validation_step, 
      log_dir=log_dir,
      lr=learning_rate,
      eta_min=eta_min,
      weight_decay=weight_decay)