import json
import pickle
import numpy as np
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



with open('FB15k237ent2textOrders.txt', 'r', encoding='utf-8') as file:
# with open('WN18RRent2textOrders.txt', 'r', encoding='utf-8') as file:
# with open('YAGO3-10ent2textOrders.txt', 'r', encoding='utf-8') as file:
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
    # np.save('WN18RREntTxtWeights.npy', target_CLS_embeddings)
    # np.save('YAGO3-10EntTxtWeights.npy', target_CLS_embeddings)

