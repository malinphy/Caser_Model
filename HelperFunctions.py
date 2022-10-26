import pandas as pd
import numpy as np 
import re

### Let's define the helper functions

def time_conv(x):

    y = pd.to_datetime(x, unit='s')
    
    return y

def data_splitter(x):

    return str(x).split()[0]


def unique_definer(df,col, flatten = True ):

    unique_items = df[col].unique()
    
    return (unique_items, len(unique_items))

def time_sorter(df,user_column, time_column):

    x = df.set_index([user_column,time_column]).sort_index().reset_index()

    return x

def sequencer_multi(df,col):

    # var = df.groupby(col).aggregate(lambda tdf: tdf.unique().tolist()) 
    var = df.groupby(col).aggregate(lambda tdf: tdf.tolist()) 

    var = var.reset_index()

    return var

def sequencer_unique(df,col):

    var = df.groupby(col).aggregate(lambda tdf: tdf.unique().tolist()) 
    # var = df.groupby(col).aggregate(lambda tdf: tdf.tolist()) 

    var = var.reset_index()

    return var

def input_label_maker(df,col,window_size, splitter): ### df should be in sequential format

    if splitter != True :
        mid_items = []
        for i in range(len(df)):
            mid_items.append(df[col][i][-window_size:])
        
        return mid_items

    if splitter == True:
        last_items = []
        for i in range(len(df)):
            last_items.append(np.array(df[col][i][-1]))

        mid_items = []
        for i in range(len(df)):
            mid_items.append(df[col][i][-window_size:-1])

        return (last_items, mid_items)

def corpus_creator(unique_set, start_index = 0):
    ### 0 padding might be need for features, hence +1 value added for dictionary index
    
    var_2enc = {i+start_index:j for i,j in enumerate(unique_set)}
    enc_2var = {j:i+start_index for i,j in enumerate(unique_set)}

    return var_2enc, enc_2var

def time_splitter(df, col):
    year = []
    month = []
    day = []

    hour = []
    min = []
    seconds = []
    for i in range(len(df)):
        date, time = str(df[col][i]).split(' ')

        x,y,z = ((date.split('-')))
        year.append(int(x))
        month.append(int(y))
        day.append(int(z))

        i,j,k = (time.split(':'))
        hour.append(int(i))
        min.append(int(j))
        seconds.append(int(k))

    return year,month, day, hour, min, seconds

def last_n_taker(df, col, N):
    #### takes last N items for each users 
    dups = {k:list(v) for k,v in df.index.groupby(df[col]).items()}
    last_n_items = []
    for i in range(1,len(dups)+1):
        last_n_items.append(np.array(dups[i][0:-N], dtype = 'int32'))
    kicks = np.concatenate(last_n_items)

    return df.drop(kicks).reset_index(drop=True)

def release_year(df, col):
    year = []
    for i in range(len(df)):
        x = df[col][i].split(' ')[-1]
        x = re.sub('\(|\)','',x)
        year.append(int(x))

    return year
    
    
def train_neg_maker(num_neg, df, col, unique_movies_rank):
    num_neg = num_neg
    neg3 = []
    ui= 0
    us = []
    for i in df[col]:
        pos_set = i
        neg2 = []
  
        for j in range(num_neg):
            neg1 = []
            for k in pos_set:
      
                neg_candidate = np.random.randint(1,len(unique_movies_rank))
                while neg_candidate in pos_set:
                    neg_candidate = np.random.randint(1,len(unique_movies_rank))
                us.append(ui)
      
                neg1.append((neg_candidate))
            neg2.append(np.array(neg1))
        ui += 1
        neg3.append(neg2)
    
    return us, neg3