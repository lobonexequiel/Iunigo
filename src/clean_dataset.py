from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import TruncatedSVD
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import re
import pandas as pd
import matplotlib.pyplot as plt
from utils import clean, dictionary
from modules.clases import Caso
from pprint import pprint
import pickle

from gensim.models import Word2Vec as w2v

with open('vectorizador_20210607_.pickle', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('dataset/auto-clean.csv')
df = df.dropna()
descripciones = df.descripcion.dropna().apply(clean.general)
responsabilidad = df['Motivo de cierre']

model_w2v = w2v(
    [i.split() for i in descripciones],
    size = 300,
    window=3,
    workers=-1,
    min_count=5,
    sg=1,
    hs=0,
    iter=100
)


casos = [Caso(*v, i)
         for i, v in enumerate(zip(descripciones, responsabilidad))
         if not re.search(r'grani*\w', v[0])]

traductor = {}
for i in responsabilidad.unique():
    if 'sin' in i:
        traductor[i] = 0
    else:
        traductor[i] = 1

for i in range(len(casos)):
    casos[i].set_resp_as_vector(traductor)
    casos[i].set_descripcion(casos[i].get_clean_descripcion())


resp = [i.get_responsabilidad() for i in casos]

casos_vec = model.transform([i.get_clean_descripcion() for i in casos])

svd = TruncatedSVD(n_components=500)
svd = svd.fit(casos_vec)
svd.explained_variance_ratio_.sum()
casos_vec2 = svd.transform(casos_vec)

def get_center(vectores):
    new_vector = [0 for i in range(300)]
    for vector in vectores:
        for i,v in enumerate(vector):
            new_vector[i] += v
    if not len(vectores):
        return new_vector
    return np.array(new_vector)/len(vectores)


casos_vec3 = []
for caso in casos:
    aux = []
    for word in caso.get_descripcion().split():
        try:
            aux.append(model_w2v.wv.get_vector(word))
        except KeyError:
            pass
    centro = get_center(aux)
    casos_vec3.append(centro)

from sklearn.decomposition import PCA

pca = PCA(n_components = 200)
pca.fit(casos_vec3)
pca.explained_variance_ratio_.sum()
x_train, x_test, y_train, y_test = train_test_split(pca.transform(casos_vec3), resp)

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test, y_test)

evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 60
xgb.config_context(verbosity=2)

param = {
    'max_depth': 6,
    'eta': .5,
    'objective': 'binary:hinge',
    'nthread': 4,
    'eval_metric': 'rmse',
}

bst = xgb.train(param, dtrain, num_round, evallist)

dpredic = bst.predict(dtest)

confusion_matrix(dpredic, y_test)
accuracy_score(dpredic, y_test)
f1_score(dpredic, y_test)
