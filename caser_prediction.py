# -*- coding: utf-8 -*-
"""empty_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WpQOXKGWxkwR23AMCU_w3lrB835CUhei
"""

from google.colab import drive
drive.mount('/content/drive/')

import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from HelperFunctions import time_conv,time_sorter,unique_definer, corpus_creator
from HelperFunctions import sequencer_multi,input_label_maker

from model import model
from eval import apk, mapk
from data_prep import train_test_maker, window_df_maker, test_negative_maker
from ast import literal_eval

TARGET_NUM = 1
batch_size = 4096
L=5
d=16
d_prime=4 
drop_ratio=0.05
num_factors = 5
# num_factors = 10
dims = num_factors
num_unique_item = 3706
num_unique_user = 6040

test_df = pd.read_csv('drive/MyDrive/Colab Notebooks/trained_models/caser_model/test_df.csv')
test_df['input_items_enc'] = [literal_eval(test_df['input_items_enc'][i]) for i in range(len(test_df['input_items_enc']))]

caser_model = model(num_unique_item,num_unique_user,num_factors,L,d_prime,d)
caser_model.load_weights('drive/MyDrive/Colab Notebooks/trained_models/caser_model/caser_model_weights.h5')

c_model_b_emb = caser_model.get_layer('b_embedding').output
c_model_W_emb = caser_model.get_layer('W_embedding').output
c_model_x = caser_model.get_layer('tf.concat_1').output

caser_model_predict= Model(inputs = [caser_model.inputs], outputs = [c_model_b_emb, c_model_W_emb,c_model_x])

def predictor(user_id):
    
    b_emb, W_emb, x = caser_model_predict([
                              tf.constant([test_df['user_id_enc'][user_id]]),
                              tf.constant([test_df['input_items_enc'][user_id]]),
                              tf.constant(np.atleast_2d(np.arange(num_unique_item)))
                                                                #    train_df['train_negatives'].astype('int') 
                                        ])
    # print(b_emb.shape)
    preds = (tf.math.reduce_sum((x*tf.squeeze(caser_model.get_layer('W_embedding')(np.atleast_2d(np.arange(num_unique_item))), axis = 0)),
                   axis=1) + tf.squeeze(caser_model.get_layer('b_embedding')(np.atleast_2d(np.arange(num_unique_item))) ))

    
    preds = np.array(preds)

    return (preds)



user_index = 0
print('PREDICTED ITEMS FOR USER',user_index,np.argsort(-predictor(user_index))[0:20])

test_df

