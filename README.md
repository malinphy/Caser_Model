# Caser_Model

A tensorflow/keras implementation of the unified embedding model from Embedding-based Retrieval in Facebook Search (https://arxiv.org/pdf/1809.07426.pdf).

Sequential recommendation system using convolutionally enhanced item embeddings and triplet loss.


Data :<br/>
----

MovieLens 1M Dataset :
https://grouplens.org/datasets/movielens/1m/
<br/>
for simplicity dataset located in my github repo :
https://raw.githubusercontent.com/malinphy/datasets/main/ml_1M/ratings.dat

File Description :
----
- data_prep.py :
- HelperFunctions : Data preparation for model training
- model.py : Caser model written with tensoflow/keras
- train.py : training file
- caser_model_weights.h5 : model weights 
- eval.py : mean average precision MAP calculation
- requirements.txt : required packages and versions to run model

Usage :
----
