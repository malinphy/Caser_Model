# Caser_Model

Tensorflow/Keras implementation of sequential recommendation system using convolutionally enhanced item embeddings and triplet loss. (https://arxiv.org/pdf/1809.07426.pdf)




Data :<br/>
----

MovieLens 1M Dataset : 
https://grouplens.org/datasets/movielens/1m/
<br/>
for simplicity dataset located in my github repo :
https://raw.githubusercontent.com/malinphy/datasets/main/ml_1M/ratings.dat

File Description :
----
- data_prep.py : generation of negative samples and target values
- HelperFunctions : Data preparation for model training
- model.py : Caser model written with tensoflow/keras
- train.py : training file
- caser_model_weights.h5 : model weights 
- eval.py : mean average precision MAP calculation
- requirements.txt : required packages and versions to run model

Usage :
----
