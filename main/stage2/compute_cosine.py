import torch
import pickle
import json

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

        embedding1 = model(file1) 