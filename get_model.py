from transformers import AutoTokenizer
import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

def get_model(model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=2).to('cuda:0')
    return tokenizer, model