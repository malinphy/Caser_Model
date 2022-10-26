#training file for caser model


# from google.colab import drive
# drive.mount('/content/drive/')

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

train_root = 'https://raw.githubusercontent.com/malinphy/datasets/main/ml_1M/ratings.dat'

df = pd.read_csv(train_root,delimiter = '::',header = None)
df.columns = ['user_id','item_id','ratings','timestamp']

df['timestamp'] = time_conv(df['timestamp'])
df = time_sorter(df,'user_id', 'timestamp')
# df.head(3)

unique_users, num_unique_user = unique_definer(df,'user_id')
unique_items, num_unique_item = unique_definer(df,'item_id')

item_2enc, enc_2item = corpus_creator(unique_items, 1)
user_2enc, enc_2user = corpus_creator(unique_users, 1)

df['user_id_enc'] = df['user_id'].map(enc_2user)
df['item_id_enc'] = df['item_id'].map(enc_2item)

df_enc = df[['user_id_enc','item_id_enc']].copy()

sequenced_df_enc = sequencer_multi(df_enc,'user_id_enc')
sequenced_df_enc.head(3)

train_seq, test_seq = train_test_maker(sequenced_df_enc,'item_id_enc',split_ratio = (0.8))
sequenced_df_enc['train_seq_enc'] = train_seq
sequenced_df_enc['test_seq_enc'] = test_seq

window_df = window_df_maker(df = sequenced_df_enc, n = 6, user_col ='user_id_enc' , sequence_col = 'train_seq_enc')

#### last entries with same user id enc will be treated as test indices
test_indices =(np.where(window_df.duplicated(subset=['user_id_enc']) == False)[0]-1)[1:]
train_df = window_df.drop(test_indices).reset_index(drop = True)
test_df = window_df.iloc[test_indices].reset_index(drop = True)

last_item_train, mid_items_train = input_label_maker(train_df,'item_id_enc',6, True)
last_item_test, mid_items_test = input_label_maker(test_df,'item_id_enc',6, True)

train_df['input_items_enc'] = mid_items_train
train_df['label_items'] = last_item_train

test_df['input_items_enc'] = mid_items_test
test_df['label_items_enc'] = last_item_test
test_df['test_set_enc'] = sequenced_df_enc['test_seq_enc']

final_test_set_enc = []
for i in range(len(test_df)):
    x = (int(test_df['label_items_enc'][i]))
    y = (test_df['test_set_enc'][i])
    y.insert(0,x)
    # final_test_set_enc.append()

train_negs = test_negative_maker(1,train_df,'item_id_enc',num_unique_item)
train_df['train_negs_enc']= train_negs

train_negs_enc = [int(i) for i in train_df['train_negs_enc']]
train_df['train_negs_enc'] = train_negs_enc

TARGET_NUM = 1
batch_size = 4096
L=5
d=16
d_prime=4 
drop_ratio=0.05
num_factors = 5
# num_factors = 10
dims = num_factors

@tf.function
def triplet_loss(y_target, y_pred):  
  pos, neg = tf.split(y_pred,2,1)
  positive_loss = -1*tf.math.reduce_mean(tf.math.log(tf.sigmoid(pos)))
  negative_loss = -1*tf.math.reduce_mean(tf.math.log(1- tf.sigmoid(neg)))
  total_loss = tf.math.add(positive_loss, negative_loss)
  return total_loss 

caser_model = model(num_unique_item,num_unique_user,num_factors,L,d_prime,d)
caser_model.compile(loss = triplet_loss,optimizer = 'Adam')

caser_model.load_weights('caser_model_weights.h5')

y_dummy = tf.ones( [len(train_df),1])
train_sequences = [np.array(i)  for i in (train_df['input_items_enc'])]
test_sequences = [np.array(i)  for i in (test_df['input_items_enc'])]

# caser_hist = caser_model.fit([
#                               tf.constant(train_df['user_id_enc']),
#                               tf.constant(train_sequences),
#                               tf.constant( np.squeeze( np.dstack( [ train_df['label_items'].astype('int'),
#                                                                    train_df['train_negs_enc'].astype('int') 
#                                                                    ] ) ) ) ],
#                 y_dummy,
#                 epochs = 20,  
#                 batch_size = 512 
#                 )

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
    # return np.array(preds)
    return (preds)

"""# EVALUATION"""

## for evaluation mean average precision is calculated for one item.
## training items were extracted from the prediction for each user.

p = []
for i in range(len(test_df)):
    p.append(list(np.argsort(-predictor(i))))

diluted_test = []
for i in range(len(p)):
    x = p[i]
    v1 = []
    for j in range(len(x)):
        if x[j] not in sequenced_df_enc['train_seq_enc'][i][0:-1] :
            v1.append(x[j])
    diluted_test.append(v1)

print('MEAN AVERAGE PRECISION',mapk(test_df['test_set_enc'], (diluted_test)))
print('end of the script')