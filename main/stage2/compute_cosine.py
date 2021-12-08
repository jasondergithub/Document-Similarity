import torch
import pickle
import json
import torch.nn as nn

import sys
sys.path.insert(1, '../stage1/')
import config
from transformers import BertForSequenceClassification

def computation():
    print(config.EPOCHS)

if __name__ == "__main__":
    computation()