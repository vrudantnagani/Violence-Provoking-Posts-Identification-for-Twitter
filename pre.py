#Importing Libraries

import tensorflow as tf
from tensorflow import keras
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#Predicting the Category for Tweets

def prediction(df1):
    d = df1.iloc[:,:]
    df1 = df1.iloc[:, 1:]
    df1.replace('[^a-zA-Z]',' ',inplace=True)
    for index in ['text']:
        df1[index]=df1[index].str.lower()

    tf1 = pickle.load(open("tfidf.pkl", 'rb'))
    tfnew=TfidfVectorizer(stop_words='english',vocabulary=tf1.vocabulary_)
    xtt=tfnew.fit_transform(df1['text'])

    def convert_sparse_matrix_to_sparse_tensor(X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
    xo=convert_sparse_matrix_to_sparse_tensor(xtt)
    xoo=tf.sparse.reorder(xo)

    model=keras.models.load_model("ann.h5")
    pr=model.predict(xoo)
    for i in range(len(pr)):
        pr[i] = round(pr[i][0])
    d['val']=pr
    return d

