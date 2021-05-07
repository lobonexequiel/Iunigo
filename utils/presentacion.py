from pprint import pprint
from nltk.stem import SnowballStemmer
import streamlit as st
import pandas as pd
import pickle
import re

def get_ubicacion_vial(descripcion):
    location = ['calle', r'garaje', r'roton\w*', 'autopista', 'avenida', 'cruce', 'cruze', r'esquina\w*',
                r'estacionami\w*', 'carril', 'ruta', r'semaforo\w*', r'intersec.?', 'tunel', 'peaje']
    aux = []
    for loc in location:
        st = re.search(loc, descripcion)
        if st:
            aux.append(st.group())
    if aux:
        return ' '.join(set(aux))
    else:
        return 'desconocido'


def dot_prod(string, num):
    aux = []
    for i in range(len(num)):
        if num[i]:
            aux.append(string[i])
    return ' '.join(aux)


lugar = ['delantera', 'trasera', 'izquierda', 'derecha']

sbs = SnowballStemmer('spanish')

# df = pd.read_csv('../dataset/dataset_final_no_dup.csv')
try:
    df = pd.read_csv('dataset/dataset_final_no_dup.csv')
    with open('models/vectorizer.pickle', 'rb') as v:
        vectorizer = pickle.load(v)

    with open('models/vectorizer1.pickle', 'rb') as v1:
        vectorizer1 = pickle.load(v1)

    with open('models/multilabel.pickle', 'rb') as ml:
        ml = pickle.load(ml)

    with open('models/svm_responsabilidad.pickle', 'rb') as r:
        svm_responsabilidad = pickle.load(r)

except:
    df = pd.read_csv('../dataset/dataset_final_no_dup.csv')

    with open('../models/vectorizer.pickle', 'rb') as v:
        vectorizer = pickle.load(v)

    with open('../models/vectorizer1.pickle', 'rb') as v1:
        vectorizer1 = pickle.load(v1)

    with open('../models/multilabel.pickle', 'rb') as ml:
        ml = pickle.load(ml)

    with open('../models/svm_responsabilidad.pickle', 'rb') as r:
        svm_responsabilidad = pickle.load(r)

st.title('''Clasificador de denuncias de siniestros''')

        
input_text = st.text_area('Breve descripcion del siniestro', height=150)

if len(input_text):
    desc_vec = vectorizer1.transform([' '.join([sbs.stem(word) for word in input_text.split()])])
    resp_pred = svm_responsabilidad.predict(desc_vec)[0]
    resp_pred = 'responasble' if bool(resp_pred) else 'no responasble'
    # resp_pred
    st.text('Responsabilidad predicha: ' + resp_pred)

    st.text('El siniestro ocurrió en: ' +
            get_ubicacion_vial(input_text))

    desc_vec = vectorizer.transform([input_text])
    lugar_pred = ml.predict(desc_vec)[0]
    st.text('El daño en el vehículo asegurado se encuentra en la parte: ' +
            dot_prod(lugar, lugar_pred))


if st.button('Caso aleatorio'):
    muestra = df.sample()
    desc = muestra.Descripción.array[0]
    st.text_area('Descripción del hecho: ',value=desc,height=150)
    # muestra['Motivo de cierre'].array[0]
    st.text('Motivo de cierre: ' + muestra['Motivo de cierre'].array[0])
    resp_base = 'responasble' if muestra['responsabilidad'].bool else 'no responasble'
    st.text('Responsabilidad: ' + resp_base)

    desc_vec = vectorizer1.transform(muestra['Descripción'].apply(lambda x: ' '.join(
        [sbs.stem(word) for word in x.split()])))
    resp_pred = svm_responsabilidad.predict(desc_vec)[0]
    resp_pred = 'responasble' if bool(resp_pred) else 'no responasble'
    # resp_pred
    st.text('Responsabilidad predicha: ' + resp_pred)

    st.text('El siniestro ocurrió en: ' +
            get_ubicacion_vial(desc))

    desc_vec = vectorizer.transform(muestra['Descripción'])
    lugar_pred = ml.predict(desc_vec)[0]
    st.text('El daño en el vehículo asegurado se encuentra en la parte: ' + dot_prod(lugar, ml.predict(desc_vec)[0]) if 1 in lugar_pred else 'no identificado')
