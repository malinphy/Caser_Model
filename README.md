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
- HelperFunctions.py : Data preparation for model training
- model.py : Caser model written with tensoflow/keras
- train.py : training file
- caser_model_weights.h5 : model weights 
- eval.py : mean average precision MAP calculation
- requirements.txt : required packages and versions to run model

Usage :
if necessary download repo and create an virtual env using following commands 
----
download file 
```
conda create --name caser_env
conda activate revenue_model
```
find the folder directory in caser_env
```
pip install -r requirements.txt 
```
run ***train.py*** file 
<br/>
for deployment purpose prediction file created seperately as ***caser_prediction.py***

TODO :
----
original study investigated up to next 3 items. However, this implementation designed for 1 item. Designing will be developed according to original model.

Citation:
----
```
If you use this Caser in your paper, please cite the paper:

@inproceedings{tang2018caser,
  title={Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding},
  author={Tang, Jiaxi and Wang, Ke},
  booktitle={ACM International Conference on Web Search and Data Mining},
  year={2018}
}

```
