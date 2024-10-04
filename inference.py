import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from tools import read_config_video, load_data
from datasets import Dataset
from get_model import get_model
import argparse
import evaluate
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import roc_curve, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("-path", help="path where weights are stored", required=True)
args = parser.parse_args()

output_folder = 'out'

config = read_config_video(os.path.join(args.path, 'config.yaml'))

# You can lower your batch size if you're running out of GPU memory
batch_size = config['batch_size']
epochs = config['epochs']
validation_step = config['validation_step']
model_checkpoint = config['model_checkpoint']
learning_rate = config['learning_rate']
eta_min = config['eta_min']
weight_decay = config['weight_decay']

_, test_sequences, _, _, test_labels, _ = load_data()

tokenizer, model = get_model(model_checkpoint)

model.load_state_dict(torch.load(os.path.join(args.path, 'weights.pth'), weights_only=True))
model.eval()

test_sequences = tokenizer(test_sequences)
test_sequences = Dataset.from_dict(test_sequences)
test_dataset = test_sequences.add_column("labels", test_labels)

data_collator = DataCollatorForTokenClassification(tokenizer)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

with torch.no_grad():
    metric = evaluate.load("accuracy")
    model.eval()
    logit_list = []
    label_list = []
    for batch in test_dataloader:
        batch = {k: v.to('cuda:0') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits # B, L, C
        predictions = torch.argmax(logits, dim=-1)

        probs = torch.softmax(logits, dim=-1)[:, :, 1]

        labels = batch["labels"].reshape((-1,))
        predictions = predictions.reshape((-1,))
        probs = probs.reshape((-1,))

        predictions = predictions[labels!=-100]
        probs = probs[labels!=-100]
        labels = labels[labels!=-100]

        logit_list.append(probs)
        label_list.append(labels)

        metric.add_batch(predictions=predictions, references=labels)

    logits = torch.cat(logit_list, dim=0).cpu().numpy()
    label = torch.cat(label_list, dim=0).cpu().numpy()

    accuracy = metric.compute()
    print(accuracy)
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(label, logits)
    auc_score = roc_auc_score(label, logits)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multiple Sequences')
    plt.legend()
    plt.savefig('Example.png', dpi=600)
    plt.show()