import torch
import pickle
import json
import torch.nn as nn

import sys
sys.path.insert(1, '../stage1/')
import config

from itertools import combinations
from transformers import BertModel

def computation():

    result_list = []

    device = torch.device(config.DEVICE)    
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    bert_model.to(device)

    with open('../../dict/first_stage_predictions.txt', 'rb') as fp:
        prediction_list = pickle.load(fp)
        
    #example = [(1001, 7), (1001, 10), (1001, 12), (1008, 100), (1008, 125), (1008, 130), (1200, 1), (1200, 6)]
    firstElement = prediction_list[0][0]
    temp = []
    ans = []
    for element in prediction_list:        
        if firstElement != element[0]:
            result = combinations(temp, 2)
            ans += list(result)
            firstElement = element[0]
            temp.clear()        
        temp.append(element[1])

    ans += list(combinations(temp, 2)) #has all the combintations

    
    for i in range(len(ans)):
        
        with open('../../public_processed_test_files/' + str(ans[i][0]) + '.txt', 'r', encoding='UTF-8') as text1:
            file1 = text1.read()

        with open('../../public_processed_test_files/' + str(ans[i][1]) + '.txt', 'r', encoding='UTF-8') as text2:
            file2 = text2.read()
        
        labels = torch.tensor([1]).unsqueeze(0)
        labels = labels.to(device)
        input1 = config.tokenizer(file1, return_tensors="pt")
        input1.to(device)
        embedding1 = bert_model(**input1)

        h1 =  embedding1.last_hidden_state
        h1 = torch.squeeze(h1, 0)
        h1_cls = h1[0]

        input2 = config.tokenizer(file2, return_tensors="pt")
        input2.to(device)
        embedding2 = bert_model(**input2)
        
        h2 =  embedding2.last_hidden_state
        h2 = torch.squeeze(h2, 0)
        h2_cls = h2[0]

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        output = cos(h1_cls, h2_cls)
        #print(output)
        if output >= 0.7:
          result_list.append(prediction_list[i])

    return result_list
    
if __name__ == "__main__":
    computation()