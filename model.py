from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
import pandas as pd
import numpy as np
import torch

class ShallowBKGC:
    def __init__(self, *, settings, num_entities, num_relations):
        self.settings = settings


        input_head = Input(shape=(1,), dtype='int32', name='input_head')
        input_tail = Input(shape=(1,), dtype='int32', name='input_tail')

        embedding_layer = Embedding(input_dim=num_entities, output_dim=self.settings['embedding_dim'],
                                    input_length=1, activity_regularizer=l2(self.settings['reg']))
        head_embedding_e = embedding_layer(input_head)
        head_embedding_drop = Dropout(self.settings['input_dropout'])(head_embedding_e)
        tail_embedding_e = embedding_layer(input_tail)
        tail_embedding_drop = Dropout(self.settings['input_dropout'])(tail_embedding_e)

        embedding_weights = np.load('FB15K237EntTxtWeights.npy', allow_pickle=True)
        #embedding_weights = np.load('WN18RREntTxtWeights.npy', allow_pickle=True)
        #embedding_weights = np.load('YAGOEntTxtWeights.npy', allow_pickle=True)
        embedding_weights_reshape = torch.stack(list(embedding_weights))
        embedding_weights = embedding_weights_reshape.numpy()
        embedding_weights_dim = embedding_weights[:, :self.settings['embedding_dim']]
        embedding_text_layer = Embedding(input_dim=len(embedding_weights_dim), output_dim=self.settings['embedding_dim'])
        embedding_text_layer.build((None,))
        embedding_text_layer.set_weights([embedding_weights_dim])
        embedding_text_layer.trainable = False

        head_embedding_e_text = embedding_text_layer(input_head)
        tail_embedding_e_text = embedding_text_layer(input_tail)

        h_embedding = Average()([head_embedding_e_text, head_embedding_drop])
        t_embedding = Average()([tail_embedding_e_text, tail_embedding_drop])

        h_embedding_dense = Dense(self.settings['embedding_dim'] * self.settings['hidden_width_rate'],
                                  activity_regularizer=l2(self.settings['reg']))(h_embedding)
        h_embedding_dense_d = Dropout(self.settings['hidden_dropout'])(h_embedding_dense)
        t_embedding_dense = Dense(self.settings['embedding_dim'] * self.settings['hidden_width_rate'],
                                  activity_regularizer=l2(self.settings['reg']))(t_embedding)
        t_embedding_dense_d = Dropout(self.settings['hidden_dropout'])(t_embedding_dense)
        combined = Average()([h_embedding_dense_d, t_embedding_dense_d])

        final_f = Flatten()(combined)
        final_d_relu = Dense(self.settings['embedding_dim'] * self.settings['hidden_width_rate'],
                             activation="relu")(final_f)
        final_d = Dense(num_relations, activation="sigmoid")(final_d_relu)

        self.model = Model(inputs=[input_head, input_tail], outputs=final_d)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        
    def fit(self,X, y):
        X_Head = np.array(X[:, 0])
        X_Tail = np.array(X[:, 1])
        len_X_Head = len(X_Head)
        len_X_Tail = len(X_Tail)
        X_Head = np.reshape(X_Head, (len_X_Head,1))
        X_Tail = np.reshape(X_Tail, (len_X_Tail,1))
        self.model.fit([X_Head, X_Tail], y, batch_size=self.settings['batch_size'], epochs=self.settings['epochs'],
                       use_multiprocessing=True, verbose=1, shuffle=True)


    def predict(self, X):
        X_Head = np.array(X[:, 0])
        X_Tail = np.array(X[:, 1])
        len_X_Head = len(X_Head)
        len_X_Tail = len(X_Tail)
        X_Head = np.reshape(X_Head, (len_X_Head, 1))
        X_Tail = np.reshape(X_Tail, (len_X_Tail, 1))
        return self.model.predict([X_Head, X_Tail])


