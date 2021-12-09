import torch
import pickle
import json
import torch.nn as nn

import sys
sys.path.insert(1, '../stage1/')
import config

from itertools import combinations
from transformers import BertForSequenceClassification

def computation():
    example = [(1001, 7), (1001, 10), (1001, 12), (1008, 100), (1008, 125), (1008, 130), (1200, 1), (1200, 6)]
    firstElement = example[0][0]
    temp = []
    ans = []
    for element in example:        
        if firstElement != element[0]:
            result = combinations(temp, 2)
            ans += list(result)
            firstElement = element[0]
            temp.clear()        
        temp.append(element[1])

    ans += list(combinations(temp, 2))
    print(ans)

if __name__ == "__main__":
    computation()