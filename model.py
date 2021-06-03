#Importing Libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
warnings.filterwarnings('ignore')

#Reading the dataset

df=pd.read_csv(r"E:/My/SRM UNIVERSITY/3rd year/Sem 6/Seminar/Model/agr_en_train.csv")
print(df)
df.columns=["user","text","type"]
df1=df.iloc[:,1:]
df1.isna().sum()
df1.groupby('type').count()
df1.replace('CAG','0',inplace=True)
df1.replace('NAG','0',inplace=True)
df1.replace('OAG','1',inplace=True)
df1.groupby('type').count()

#Data Cleaning

df1.replace('[^a-zA-Z]',' ',inplace=True)
for index in ['text','type']:
    df1[index]=df1[index].str.lower()

#TF-IDF Vectorization

t=TfidfVectorizer(stop_words='english')
xt=t.fit_transform(df1['text'])
pickle.dump(t,open("tfidf.pkl","wb"))

yy=df1['type'].to_numpy()

#Over Sampling the Required Dataset

from imblearn.over_sampling import RandomOverSampler
os= RandomOverSampler()
x,y=os.fit_resample(xt,yy)
from collections import Counter
print(Counter(y))
X1=x[:16000]
X2=x[16000:]
Y1=y[:16000]
Y2=y[16000:]

#Training with Logistic-Regression Model

l=LogisticRegression()
ml=l.fit(X1,Y1)
ml.score(X2,Y2)

filename = 'E:/My/SRM UNIVERSITY/3rd year/Sem 6/Seminar/Model/lr_model.sav'
pickle.dump(ml, open(filename, 'wb'))

# Creating ANN Model

np.random.seed(42)
tf.random.set_seed(42)

inputA= keras.Input(shape=(X1.shape[1]),sparse=True)
x= keras.layers.Dense(20, activation="relu")(inputA)
x= keras.layers.Dense(40, activation="relu")(x)
x= keras.layers.Dense(64, activation="relu")(x)
x= keras.layers.Dense(40, activation="relu")(x)
x= keras.layers.Dense(20, activation="relu")(x)
x= keras.layers.Dense(1,activation='sigmoid')(x)
x= keras.Model(inputs=inputA, outputs=x)
x.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Converting from sparse matrix to sparse tensor

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

# Training the ANN Model

xo=convert_sparse_matrix_to_sparse_tensor(X1)
xoo=tf.sparse.reorder(xo)
Y1=np.asarray(Y1).astype('float32')
x.fit(xoo,Y1,epochs=50)

keras.models.save_model(x,"ann.h5")

# Calculating the accuracy

x22=convert_sparse_matrix_to_sparse_tensor(X2)
xx2=tf.sparse.reorder(x22)

pr=x.predict(xx2)


Y2=np.asarray(Y2).astype('float32')
for i in range(len(pr)):
    pr[i]=round(pr[i][0])

sum=0
for i in range(len(pr)):
    if(pr[i][0]==Y2[i]):
        sum+=1
print(sum/len(Y2))