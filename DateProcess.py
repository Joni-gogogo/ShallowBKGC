import json
import pickle
import numpy as np



'''
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
'''

from pytorch_pretrained_bert import BertTokenizer, BertModel
import logging
import pandas as pd
import numpy as np
import torch

# loading the pre-trained Bert model
model = BertModel.from_pretrained('bert-base-uncased')
# put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Funtion to massage the input into the right form
def bert_text_preparation(text, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)
    # convert inputs to pytorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokenized_text, tokens_tensor, segments_tensors
# Funtion to convert the input into embeddings
def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    with torch.no_grad():
        _, pooled_output = model(tokens_tensor, segments_tensors)
    # collapsing the tensor into 1-dimension
    pooled_output_emb = torch.squeeze(pooled_output, dim=0)
    # converting torch tensors to lists
    # list_pooled_outputs = [pooled_embed.tolist() for pooled_embed in pooled_output_emb]
    return pooled_output_emb


'''
# Embeddings are generated in the following manner
target_CLS_embeddings = []

#f_ent2textOrders = open('WN18ent2textOrders.txt','r',encoding='utf-8')
#f_ent2textOrders = open('FB15k237ent2textOrders.txt','r',encoding='utf-8')

#line_count = sum(1 for line in f_ent2textOrders)
#print("Total lines in the file:", line_count) 

count = 0
for text in f_ent2textOrders:
    text = text.strip('\n')
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
    pooled_output_emb = get_bert_embeddings(tokens_tensor, segments_tensors, model)
    target_CLS_embeddings.append(pooled_output_emb)
    count = count + 1
    print(count)
print(len(target_CLS_embeddings))

target_CLS_embeddings = np.array(target_CLS_embeddings)
np.save('FB15K237EntTxtWeights.npy',target_CLS_embeddings)
print(target_CLS_embeddings.shape)
'''

#embedding_weights = np.load('FB15k237EntTxtWeights.npy', allow_pickle=True)
#print(embedding_weights.shape)

with open('FB15k237ent2textOrders.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for idx, line in enumerate(lines):
        try:
            processed_line = line.strip('\n')
            tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(processed_line, tokenizer)
            pooled_output_emb = get_bert_embeddings(tokens_tensor, segments_tensors, model)
            target_CLS_embeddings.append(pooled_output_emb)

        except Exception as e:
            print(f"Error in line {idx + 1}: {e}")
            print(f"Problematic line: {line}")


    target_CLS_embeddings = np.array(target_CLS_embeddings)
    np.save('FB15K237EntTxtWeights.npy', target_CLS_embeddings)
    print(target_CLS_embeddings.shape)
