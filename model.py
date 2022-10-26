### caser model for tensorflow implementation
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *


def model(num_unique_item,num_unique_user,num_factors,L,d_prime,d):
    dims = num_factors
    def expander1(x):
        return (x[:,:,:,tf.newaxis])
    
    def seqer2(x):
        return (tf.squeeze(x, axis=2))

    def seqer1(x):
        return (tf.squeeze(x, axis=1))

    user_input = tf.keras.Input(shape = (1,), name = 'user_input') 
    seq_input = tf.keras.Input(shape = (L,), name = 'seq_input') 

    seq_embedding = Embedding(num_unique_item+1,num_factors,name = 'sequence_embedding')
    seq_embedding_layer = seq_embedding(seq_input)
    seq_embedding_l = tf.expand_dims(seq_embedding_layer, 3,name = 'sequence_embedding_expander')
    user_embedding = Embedding(num_unique_user+1, num_factors,name = 'user_embedding')
    user_embedding_layer = user_embedding(user_input)   

    h = [i + 1 for i in range(L)]
    fc1_dim_v = d_prime * num_factors
    # out_v = Conv2D(4,(L,1), name = 'convo_v')(seq_embedding_l)
    out_v = Conv2DTranspose(d_prime,(L,1), name = 'convo_v')(seq_embedding_l)
    out_v = Flatten(name = 'flatten_convo_v')(out_v)
    out_hs = []
    for i in (h):
        seq_mod = tf.keras.Sequential([
            Conv2D(d, (i, num_factors)),
            # Conv2D(16, (i, num_factors)),
            Lambda(seqer2),
            MaxPool1D(L - i + 1),
            Lambda(seqer1)                      
            ])
    out_hs.append(seq_mod(seq_embedding_l))
    out_h = tf.concat(out_hs, axis=1, name ='sequential_out') 
    out = tf.concat([out_v, out_h], axis = 1, name ='convo_concat')
    dropout_layer = Dropout(0.5, name = 'dropout_layer')(out)
    z = Dense(dims, name = 'dense_layer', activation = 'relu')(dropout_layer)
    flat_users = Flatten(name = 'flatten')(user_embedding_layer)
    x = tf.concat([z, flat_users], axis=1, name = 'concat_layer_x')    
    
    doublet_input = tf.keras.Input(shape = (None,),dtype=tf.int64, name = 'doublet_input')
    W = Embedding(num_unique_item+1, num_factors*2,name = 'W_embedding')
    b_embedding = Embedding(num_unique_item+1, 1,name = 'b_embedding')
    W_layer = W(doublet_input)## buyuk W
    b_embedding_layer = b_embedding(doublet_input)## kucuk b  
    #### step2
    res = tf.matmul(W_layer,tf.expand_dims(x,axis=-1))                         
    res = res+b_embedding_layer
    res = tf.squeeze(res,axis =2)  

    return Model(inputs= [user_input,seq_input,doublet_input],outputs = res)   