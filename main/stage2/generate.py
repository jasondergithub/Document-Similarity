import pickle

from compute_cosine import computation

with open('../../dict/first_stage_predictions.txt', 'rb') as fp:
    prediction_list = pickle.load(fp)

temp = []
tempResult = []
firstElement = prediction_list[0][0]
#example = [(1001, 7), (1001, 10), (1001, 12), (1008, 100), (1008, 125), (1008, 130), (1200, 1), (1200, 6)]
for element in prediction_list:
    if firstElement != element[0]:
        compList = computation(temp)
        for result in compList:
            tempResult.append((result[0], firstElement))
            tempResult.append((result[1], firstElement))            
        temp.clear()
        firstElement = element[0]
    temp.append(element[1])