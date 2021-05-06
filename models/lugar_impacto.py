from sklearn.multioutput import MultiOutputClassifier
from sklearn.semi_supervised import SelfTrainingClassifier as stc
from sklearn.calibration import CalibratedClassifierCV
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# from sklearn.metrics import pairwise_distances, silhouette_score
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.decomposition import PCA, TruncatedSVD
# from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
# import re
# import matplotlib.pyplot as plt
import nltk
import numpy as np
# import os
import pandas as pd
import pickle
# import random
# import seaborn as sns
# import sys
# import time
import functools

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from nltk.stem import SnowballStemmer
sbs = SnowballStemmer('spanish')
stops = nltk.corpus.stopwords.words('spanish')
with open('vectorizer_20210504.pickle', 'rb') as f:
    vectorizer = pickle.load(f)


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
    except Exception:
        return -1


df = pd.read_csv('../dataset/auto_full_20210504.csv').drop(columns='Unnamed: 0')
df = df[['descripcion', 'punto de impacto en asegurado']]
print(df)
# Aca tomamos una muestra balanceada, es casi la mitad de los casos clasificado
#df2test = df.groupby('punto de impacto en asegurado')
# df2test = df2test.apply(lambda x: x.sample(int(df2test.size().min()/3)))
# df2test_index = [i[1] for i in df2test.index]

#df = df.drop(df2test_index).reset_index(drop=True)

df1 = pd.read_csv('../dataset/dataset_final_20210504.csv').rename(
    columns={'Descripción': 'descripcion'})

print(df1)
df_merged = df.append(df1)
df_merged = df_merged.drop_duplicates('descripcion').reset_index(drop=True)

"""
TODO:
Para mejorar, cambiar el nombre terminado en 'o' por 'o' en el dataset
original así hacemos un for
"""

parcial = functools.partial(lugar, lugar='izquierdo')
df_merged['izquierda'] = df_merged['punto de impacto en asegurado'].apply(
    parcial)
#df2test['izquierda'] = df2test['punto de impacto en asegurado'].apply(parcial)

parcial = functools.partial(lugar, lugar='derecho')
df_merged['derecha'] = df_merged['punto de impacto en asegurado'].apply(
    parcial)
#df2test['derecha'] = df2test['punto de impacto en asegurado'].apply(parcial)

parcial = functools.partial(lugar, lugar='frontal')
df_merged['delantera'] = df_merged['punto de impacto en asegurado'].apply(
    parcial)
#df2test['delantera'] = df2test['punto de impacto en asegurado'].apply(parcial)

parcial = functools.partial(lugar, lugar='trasero')
df_merged['trasera'] = df_merged['punto de impacto en asegurado'].apply(
    parcial)
#df2test['trasera'] = df2test['punto de impacto en asegurado'].apply(parcial)

df2test = df_merged.sample(500)
df_merged = df_merged.drop(df2test.index).reset_index(drop=True)
df2test.reset_index(drop=True, inplace=True)

df_merged['descripcion'] = df_merged['descripcion'].apply(
    lambda x: ' '.join([sbs.stem(word) for word in x.split()]))

pos_mat = np.array(df_merged[['delantera', 'trasera', 'izquierda', 'derecha']])

svc = svm.SVC(C=1.5)
svc_calibrated = CalibratedClassifierCV(base_estimator=svc)
stc_model = stc(svc_calibrated)
multilabel_classifier = MultiOutputClassifier(stc_model, n_jobs=4)
y_train = pos_mat
x_train = vectorizer.transform(df_merged['descripcion'])

multilabel_classifier.fit(x_train, y_train)

x_test = vectorizer.transform(df2test['descripcion'].apply(
    lambda x: ' '.join([sbs.stem(word) for word in x.split()])))
y_test_pred = multilabel_classifier.predict(x_test)


#x_train = vectorizer.transform(df2test['descripcion'])
# ESTO HAY QUE CORRER DE NUEVO PORQUE NUNCA FUE STEMIZADO


#df_merged['descripcion'] = df_merged['descripcion'].apply(
#    lambda x: ' '.join([sbs.stem(word) for word in x.split()]))
# df_merged['descripcion']
# for parte in ['izquierda', 'derecha', 'delantera', 'trasera']:
#  #   y_train = df2test[parte]
#     #    svc_calibrated.fit(x_train, y_train)
#     stc_model = stc(svc_calibrated)
#     y = np.array(df_merged[parte])
#     x = vectorizer.transform(df_merged['descripcion'])
#     print('fitting model', parte)
#     stc_model.fit(x, y)
#     with open('models/stc_model_' + parte+'_20210502_balanced.pickle', 'wb') as f:
#         pickle.dump(stc_model, f)

"""
    Aca es donde tenemos que predecir sobre el df2test
    """
# x_test = vectorizer.transform(df2test['descripcion'])
# y_test = np.array(df2test[parte])
# y_pred = stc_model.predict(x_test)
# # y_pred = stc_model.predict(x_test)])
# # df['trasera_pred'] = pd.Series(stc_model.predict(x))
# # df[['descripcion', 'trasera_pred']]
# # df.to_csv('trasera_clasificado_test.csv', index=False)
# print(parte)
# print('F1:', f1_score(y_test, y_pred))
# print('Precisión: ', accuracy_score(y_test, y_pred))
# print('Matriz de confusión: ')
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# print('tn:', tn)
# print('tp:', tp)
# print('fn:', fn)
# print('fp:', fp)
# print(confusion_matrix(y_test, y_pred))

with open('multilabel_20210504.pickle', 'wb') as ml:
    pickle.dump(multilabel_classifier, ml)

df2test['lugar_predicted'] = pd.Series(pd.Series([i for i in y_test_pred]))
df2test.to_csv('dataset_final_para_validar_sim_20210405.csv')
