from sklearn.semi_supervised import SelfTrainingClassifier as stc
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import random
import seaborn as sns
import sys
import time
import functools

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from tqdm import tqdm as tq

import spacy

from nltk.stem import SnowballStemmer
sbs = SnowballStemmer('spanish')


nlp = spacy.load("es_core_news_sm")
stops = nltk.corpus.stopwords.words('spanish')


def lugar(string, lugar):
    """
    retorna 1 si el lugar de impacto esta en el string 0 sino

    :param string: etiqueta lugar del impacto
    :type string: string
    :param lugar: lugar a verificar si esta o no
    :tpye lugar: string

    """
    try:
        if lugar in string.lower():
            return 1
        else:
            return 0
    except:
        return -1


df = pd.read_csv('dataset/auto_full.csv')
df = df.drop(columns='Unnamed: 0')
df = df[['descripcion', 'punto de impacto en asegurado']]
df2test = df.sample(100)
index_delete = df2test.index
df = df.drop(index_delete)
df.reset_index(drop=True, inplace=True)
df1 = pd.read_csv('dataset/dataset_final_no_dup.csv')
df1 = df1.rename(columns={'Descripción': 'descripcion'})
df1 = df1.drop(columns='lugar_impacto')
df1.columns
df_merged = df.append(df1)
df_merged.columns
df_merged = df_merged.drop_duplicates('descripcion').reset_index(drop=True)


# parcial = functools.partial(lugar, lugar='izquierdo')
# df['izquierda'] = df['punto de impacto en asegurado'].apply(parcial)
# parcial = functools.partial(lugar, lugar='derecho')
# df['derecha'] = df['punto de impacto en asegurado'].apply(parcial)
# parcial = functools.partial(lugar, lugar='frontal')
# df['delantera'] = df['punto de impacto en asegurado'].apply(parcial)
"""
NOTA: el x y y que voy a hacer hasta entrenar el modelo es el de entrenemiento
para trasera
"""
parcial = functools.partial(lugar, lugar='trasero')
df_merged['trasera'] = df_merged['punto de impacto en asegurado'].apply(
    parcial)

svc = svm.SVC(C=1.5)
svc_calibrated = CalibratedClassifierCV(base_estimator=svc)


"""
tenemos que encontrar casos donde el punto de impacto no este definido
así podemos agregar eso al dataset para que reconozca cuando no lo esta
"""
# for i in range(len(df)):
#     if sum(df.loc[i,['izquierda','derecha','delantera','trasera']]) == 0:
#         print(df.iloc[i])

#col_nro = 2
#print('Entrenando con la columna: ', df.columns[-col_nro])
# print(df[df.columns[-col_nro]].value_counts())

#y = np.array(df[df.columns[-col_nro]])
y = np.array(df_merged['trasera'])

#y_labeled = np.array(df[df.columns[-col_nro]][df[df.columns[-col_nro]] != -1])
#y_unlabel = np.array(df[df.columns[-col_nro]][df[df.columns[-col_nro]] == -1])

with open('models/vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)
stc_model = stc(svc_calibrated)


x = vectorizer.transform(df_merged['descripcion'])

# x_train, x_test, y_train, y_test = train_test_split(
#    x, y, test_size=0.2, random_state=1)

stc_model.fit(x, y)
# with open('models/stc_model_trasera.pickle','wb') as f:
#    pickle.dump(stc_model,f)
"""
Aca es donde tenemos que predecir sobre el df2test
"""
x_test = vectorizer.transform(df2test['descripcion'])
df2test['trasera'] = df2test['punto de impacto en asegurado'].apply(parcial)
y_test = np.array(df2test['trasera'])
y_pred = stc_model.predict(x_test)
# y_pred = stc_model.predict(x_test)])
# df['trasera_pred'] = pd.Series(stc_model.predict(x))
# df[['descripcion', 'trasera_pred']]
# df.to_csv('trasera_clasificado_test.csv', index=False)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# print('Training MLP..')
# mlp = MLPClassifier(hidden_layer_sizes=(100,4,2),max_iter=300,learning_rate='invscaling', random_state=1)
# mlp.fit(x_train,y_train)

# with open('mlp_trasera.pickle','wb') as f:
#     pickle.dump(mlp,f)

# print(f1_score(y_test,mlp.predict(x_test)))
# print(mlp.score(x_test,y_test))

# print('Training SVC..')

# svm = svm.SVC(C=1.5)
# svm.fit(x_train, y_train)
# df['svc_delantera_pred'] = svm.predict(x)
# with open('svc_trasera.pickle','wb') as f:
#     pickle.dump(svm,f)


# y_pred = svm.predict(x_test)
# print(f1_score(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
# df.to_csv('delantera_pred-svc_test.csv', index=False)
