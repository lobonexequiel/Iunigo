from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
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
import random
import seaborn as sns
import sys
import time

from sklearn.neural_network import MLPClassifier

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from tqdm import tqdm as tq

import spacy

from nltk.stem import SnowballStemmer
sbs = SnowballStemmer('spanish')


nlp = spacy.load("es_core_news_sm")
stops = nltk.corpus.stopwords.words('spanish')
stops = [i for i in stops if i != 'mi']
stops = [j for j in stops if j != 'su']
"""
Descomentar las posibilidades o cambiar el nombre de los archivos a cargar
"""

# df = pd.read_excel('../../dataset/denuncias_iunigo_20210330.xlsx')
df = pd.read_csv('../dataset/dataset_final_20210504.csv', encoding='latin-1')
print(df)
to_del = []
for i, v in enumerate(df):
    if 'granizo' in df.loc[i, 'descripcion'].lower() or 'cleas' in df.loc[i, 'descripcion'].lower():
        to_del.append(i)
df = df.drop(df.index[to_del])
# df.dropna(inplace=True)
print('Cleanning..')

desc = df['descripcion'].apply(lambda x: ' '.join(
    [sbs.stem(word) for word in x.split()]))
print('Vectorizing..')

"""
Instanciamos el vectorizador que más nos parezca 
con ambos 2 tipos de vectorizador dio resultados similares
pero le confío más al tfidf
"""

vectorizer = TfidfVectorizer(stop_words=stops, ngram_range=(1, 5), min_df=5)
# vectorizer = CountVectorizer(stop_words=stops,ngram_range=(1,6))
vectorizer.fit(desc)

with open('vectorizer_20210504.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)

desc_vec = vectorizer.transform(desc)
"""
Los intentos de reducir la dimensión para mejorar la clasificación
fueron en vano pero dejo el código para saber que use y probar
con otra cosa
"""
# print('Dimension reduction..')
# tSVD = TruncatedSVD(n_components=500,random_state=1,n_iter=100)
# desc_vec = tSVD.fit_transform(desc_vec)

# la responsabilidad puede ser un vector binario o no
resp_vec = df['responsabilidad']
print(len(resp_vec))
"""
el perceptron es sensible a la escala de los valores por lo que vamos a utilizar un escalador
escalamos sobre todos los datos
TODO: después probar fittear sobre el train set y escalar con ese fit 
TODO: encontrar un scaler para sparse 
NO FUNCIONÓ
"""
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(desc_vec)
# desc_vec_scaled = scaler.transform(desc_vec)

x_train, x_test, y_train, y_test = train_test_split(
    desc_vec, resp_vec, test_size=0.25)
print(x_train.shape)
# mlp = MLPClassifier(hidden_layer_sizes=(30, 4, 2), random_state=1,
#                     max_iter=500, activation='relu', learning_rate='invscaling')
# mlp.fit(x_train, y_train)

# Descomentar para guardar el modelo entrenado
# with open('mlp_1.pickle','wb') as var:
#     pickle.dump(mlp,var)

print('Training..')
# hidden_layer_sizes: el elemento i-esimo hace referencia a la cantidad de neuronas de la capa i-esima
# mlp = MLPClassifier(hidden_layer_sizes=(30, 4, 2), random_state=1,
#                     max_iter=500, activation='relu',
#                     learning_rate='invscaling')
# mlp.fit(x_train, y_train)

# with open('mlp_responsabilidad.pickle', 'wb') as var:
#     pickle.dump(mlp, var)


# print('Score: ',mlp.score(x_test,y_test))

# y_pred = mlp.predict(x_test)

# print('MLP')

# print('F1 score: ', f1_score(y_test, y_pred))
# print('Accuaracy score:', accuracy_score(y_test, y_pred))


svm_clf = svm.SVC(C=1.5)
svm_clf.fit(x_train, y_train)

# Descomentar para guardar el modelo entrenado
with open('svm_responsabilidad_20210504.pickle', 'wb') as f:
    pickle.dump(svm_clf, f)

y_pred = svm_clf.predict(x_test)

print("SVM")

print('F1 score: ', f1_score(y_test, y_pred))
print('Accuaracy score:', accuracy_score(y_test, y_pred))
