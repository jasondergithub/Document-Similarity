import torch
import pickle
import json
import torch.nn as nn

from stage1 import config
from transformers import BertForSequenceClassification

def computation():
    device = torch.device(config.DEVICE)
    model =  BertForSequenceClassification(config.bert_config)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with open('../../dict/first_stage_predictions.txt', 'rb') as fp:
        prediction_list = pickle.load(fp)
    
    for i in range(len(prediction_list)):
        
        with open('../../public_processed_test_files/' + str(prediction_list[i][0]) + '.txt', 'r', encoding='UTF-8') as text1:
            file1 = text1.read()

        with open('../../public_processed_test_files/' + str(prediction_list[i][1]) + '.txt', 'r', encoding='UTF-8') as text2:
            file2 = text2.read()
        
        labels = torch.tensor([1]).unsqueeze(0)

        input1 = config.tokenizer(file1, return_tensors="pt")
        embedding1 = model(**input1, labels=labels, output_hidden_states=True)
        h1 =  embedding1.hidden_states[-1]

        input2 = config.tokenizer(file2, return_tensors="pt")
        embedding2 = model(**input2, labels=labels, output_hidden_states=True)
        h2 =  embedding2.hidden_states[-1]

        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        output = cos(input1, input2)

if __name__ == "__main__":
    computation()