from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier
from sklearn.semi_supervised import SelfTrainingClassifier as stc
from sklearn.calibration import CalibratedClassifierCV
import nltk
import numpy as np
import pandas as pd
import pickle
import functools

from nltk.stem import SnowballStemmer


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
        return 0
    except Exception as e:
        return -1


sbs = SnowballStemmer('spanish')
stops = nltk.corpus.stopwords.words('spanish')

with open('vectorizer_20210504.pickle', 'rb') as f:
    vectorizer = pickle.load(f)


df = pd.read_csv(
    '../dataset/auto_full_20210504.csv').drop(columns='Unnamed: 0')
df = df[['descripcion', 'punto de impacto en asegurado']]

df1 = pd.read_csv('../dataset/dataset_final_20210504.csv').rename(
    columns={'Descripción': 'descripcion'})

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

parcial = functools.partial(lugar, lugar='derecho')
df_merged['derecha'] = df_merged['punto de impacto en asegurado'].apply(
    parcial)

parcial = functools.partial(lugar, lugar='frontal')
df_merged['delantera'] = df_merged['punto de impacto en asegurado'].apply(
    parcial)

parcial = functools.partial(lugar, lugar='trasero')
df_merged['trasera'] = df_merged['punto de impacto en asegurado'].apply(
    parcial)

df2test = df_merged.sample(500)
df_merged = df_merged.drop(df2test.index).reset_index(drop=True)
df2test.reset_index(drop=True, inplace=True)

df_merged['descripcion'] = df_merged['descripcion'].apply(
    lambda x: ' '.join([sbs.stem(word) for word in x.split()]))

pos_mat = np.array(df_merged[['delantera', 'trasera', 'izquierda', 'derecha']])

svc = svm.SVC(C=1.5)
svc_calibrated = CalibratedClassifierCV(base_estimator=svc)  # esto es en vano?
stc_model = stc(svc_calibrated)

multilabel_classifier = MultiOutputClassifier(stc_model, n_jobs=4)

y_train = pos_mat
x_train = vectorizer.transform(df_merged['descripcion'])

multilabel_classifier.fit(x_train, y_train)

x_test = vectorizer.transform(df2test['descripcion'].apply(
    lambda x: ' '.join([sbs.stem(word) for word in x.split()])))
y_test_pred = multilabel_classifier.predict(x_test)


# x_train = vectorizer.transform(df2test['descripcion'])
# ESTO HAY QUE CORRER DE NUEVO PORQUE NUNCA FUE STEMIZADO


with open('multilabel_20210504.pickle', 'wb') as ml:
    pickle.dump(multilabel_classifier, ml)

df2test['lugar_predicted'] = pd.Series([i for i in y_test_pred])

# comento esto porque hay que tener cuidado a la hora de correr todo
# df2test.to_csv('dataset_final_para_validar_sim_20210405.csv')
