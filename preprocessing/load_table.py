import pickle
import random
import json

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return len(lst3)


with open("../dict/relatedTable.txt", "rb") as fp:
    related_table = pickle.load(fp)

with open("../dict/unrelatedTable.txt", "rb") as fp:
    unrelated_table = pickle.load(fp)

name_list = ['table1.txt', 'table2.txt', 'table3.txt', 'table4.txt', 'table5.txt', 'table6.txt', 
'table7.txt', 'table8.txt', 'table9.txt', 'table10.txt', 'table11.txt', 'table12.txt', 'table13.txt', 'table14.txt',
'table15.txt', 'table16.txt', 'table17.txt', 'table18.txt', 'table19.txt', 'table20.txt']
# for i in range(20):
#     if (i+1) % 5 == 0:
#         subtable = random.sample(unrelated_table, 2762)
#     else:
#         subtable = random.sample(unrelated_table, 1381)    
#     subtable = subtable + related_table
#     random.shuffle(subtable)
#     with open("../table/" + name_list[i], "wb") as fp:
#         pickle.dump(subtable, fp)

# subtable = random.sample(unrelated_table, 6905) #8286
# for i in range(4): #4
#     subtable += related_table

# random.shuffle(subtable)
# with open("../table/table100.txt",  "wb") as fp:
#     pickle.dump(subtable, fp)

tf = open('../dict/keywordSet.json', 'r')
keywordSet = json.load(tf)
keys_list = list(keywordSet)

temp1 = []
temp2 = []
encode_train_list = []

for i in range(len(unrelated_table)):
    with open('../processed_files/' + str(unrelated_table[i][0]) + '.txt', 'r', encoding='UTF-8') as text1:
        file1 = text1.read()

    with open('../processed_files/' + str(unrelated_table[i][1]) + '.txt', 'r', encoding='UTF-8') as text2:
        file2 = text2.read() 

    for key in keys_list:
        value = keywordSet[key]

        if file1.find(key) != -1:
            if value not in temp1:
                temp1.append(value)
        if file2.find(key) != -1:
            if value not in temp2:
                temp2.append(value)   

    encode_train_list.append((temp1, temp2))
    temp1 = []
    temp2 = []

filter_table = []
for i in range(len(unrelated_table)):
  num = intersection(encode_train_list[i][0], encode_train_list[i][1])

  if len(encode_train_list[i][0]) != 0:
    if num/len(encode_train_list[i][0]) > 0.8:
      filter_table.append(unrelated_table[i])

# with open("../dict/filter_train.txt",  "wb") as fp:
#     pickle.dump(filter_table, fp) 

# print(len(filter_table))

subtable = related_table + filter_table
print(len(subtable))
random.shuffle(subtable)
with open("../table/table101.txt",  "wb") as fp:
    pickle.dump(subtable, fp)