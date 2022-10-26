#OTHER IMPORT FUNCTIONS FOR CASER MODEL
import numpy as np 
import pandas as pd 
def train_test_maker(df,col,split_ratio):
    train_seq = []
    test_seq = []
    for i in range(len(df)):
        x = (len(df[col][i]))
        train_ratio = int(np.round(x*(split_ratio)))
        test_ratio  = np.round(x*(1-(split_ratio)))
        # print(train_ratio)
        train_seq.append(df[col][i][:train_ratio])
        test_seq.append(df[col][i][train_ratio:])

    return(train_seq, test_seq)

def window_df_maker(df, n, user_col, sequence_col):
    n = n
    var2 =[]
    u_id = [] ### user_index
    for i in range(len(df)):
        x = df[sequence_col][i]
        var1 = []
        var_u1 = []
        for j in range(len(x)):
            batch = x[j:j+n]
            if len(batch) == n :
                var1.append(batch)
                u_id.append(df[user_col][i])
        var2.append(var1)
    var2 = np.concatenate(var2)
    var2 = [list(i) for i in var2]

    return pd.DataFrame({'user_id_enc':u_id, 'item_id_enc':var2})

def test_negative_maker(num_neg,df,col,num_unique_items):
    num_neg = num_neg
    neg3 = []
    ui= 0
    us = []
    neg2 = []
    for i in df['item_id_enc']:
        pos_set = i
  
        neg1 = []
        for j in range(num_neg):
    
            neg_candidate = np.random.randint(1,num_unique_items)

            while neg_candidate == pos_set:

                neg_candidate = np.random.randint(1,num_unique_items)
            us.append(ui)
      
            neg1.append((neg_candidate))
        neg2.append(np.array(neg1))

    return neg2