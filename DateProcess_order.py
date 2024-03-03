import json
import pickle
import numpy as np



# get FB15k237ent2textOrders.txt
f3 = open('FB15K237entityIDx_json','r')
entitydict_data = json.load(f3)
with open('FB15k237entity.txt','w') as f: 
    for key,value in entitydict_data.items(): 
        f.write(key)
        f.write('\n')



# text to dictionary
dic_temp=[]
file = open('FB15k237entity2text.txt','r', encoding='utf-8')

for line in file.readlines():
    line = line.strip('\n')
    b = line.split('\t')
    dic_temp.append(b)
dic = dict(dic_temp)

file.close()
# print(dict)
# print(len(dic))
# print(dic['/m/017dcd'])

file_EntityText = open('FB15k237ent2textOrders.txt','w',encoding='utf-8')
forder = open('FB15k237entity.txt','r',encoding='utf-8')


for l in forder.readlines():
    l = l.strip('\n')
    #value = dic[l]
    value = dic.get(l, "unknown")
    file_EntityText.write(value)
    file_EntityText.write('\n')


# get WN18RRent2textOrders.txt
'''
f3 = open('WN18RRentityIDx_json','r')
entitydict_data = json.load(f3)
with open('WN18RRentity.txt','w') as f: 
    for key,value in entitydict_data.items(): 
        f.write(key)
        f.write('\n')



# text to dictionary
dic_temp=[]
file = open('WN18RRentity2text.txt','r', encoding='utf-8')

for line in file.readlines():
    line = line.strip('\n')
    b = line.split('\t')
    dic_temp.append(b)
dic = dict(dic_temp)

file.close()
# print(dict)
# print(len(dic))
# print(dic['/m/017dcd'])

file_EntityText = open('WN18RRent2textOrders.txt','w',encoding='utf-8')
forder = open('WN18RRentity.txt','r',encoding='utf-8')


for l in forder.readlines():
    l = l.strip('\n')
    #value = dic[l]
    value = dic.get(l, "unknown")
    file_EntityText.write(value)
    file_EntityText.write('\n')
'''

# get YAGO3-10ent2textOrders.txt
'''
f3 = open('YAGO3-10entityIDx_json','r')
entitydict_data = json.load(f3)
with open('YAGO3-10entity.txt','w') as f:
    for key,value in entitydict_data.items():
        f.write(key)
        f.write('\n')



# text to dictionary
dic_temp=[]
file = open('YAGO3entity2text.txt','r', encoding='utf-8')

for line in file.readlines():
    line = line.strip('\n')
    b = line.split('\t')
    dic_temp.append(b)
dic = dict(dic_temp)

file.close()
# print(dict)
# print(len(dic))
# print(dic['/m/017dcd'])

file_EntityText = open('YAGO3-10ent2textOrders.txt','w',encoding='utf-8')
forder = open('YAGO3-10entity.txt','r',encoding='utf-8')


for l in forder.readlines():
    l = l.strip('\n')
    #value = dic[l]
    value = dic.get(l, "unknown")
    file_EntityText.write(value)
    file_EntityText.write('\n')
'''