import re
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import sys
from nltk.stem import SnowballStemmer


def dot_prod(string, num):
    aux = []
    for i in range(len(num)):
        if num[i]:
            aux.append(string[i])
    return ' '.join(aux)


def aux_fun(x):
    if not bool(len(x)):
        return 'no especificado'
    return x


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


    

# Cargamos los modelos

sbs = SnowballStemmer('spanish')
sys.path.append('~/Documentos/LegalHub/Iunigo/models/')

with open('models/vectorizer1.pickle', 'rb') as f:
    vectorizer = pickle.load(f)
with open('models/vectorizer.pickle', 'rb') as f:
    vectorizer_dir = pickle.load(f)

with open('models/svm_responsabilidad.pickle', 'rb') as var:
    svm = pickle.load(var)

with open('multilabel.pickle', 'rb') as ml:
    multilabel = pickle.load(ml)


df = pd.read_excel('dataset/dataset_final_no_dup.xlsx')
df2test = df.sample(500)
df = df.drop(df2test.index)
desc = []
for i in df2test['Descripción']:
    if type(i) == str:
        desc.append(' '.join([sbs.stem(word) for word in i.split()]))
df2test
desc_vec = vectorizer.transform(desc)
# Esto se puede hacer más rápido haciendo un diccionario de predictores
df2test['responsabilidad_pred'] = pd.Series(list(map(svm.predict, desc_vec)))
# pd.Series(list(map(delantera.predict,desc_vec)))

desc_vec_dir = vectorizer_dir.transform(df2test['Descripción'].apply(str))


df2test.reset_index(drop=True, inplace=True)
df2test.drop(columns=['Motivo de cierre', 'responsabilidad',
             'responsabilidad_pred'], inplace=True)

lugar = ['delantera', 'trasera', 'izquierda', 'derecha']

df2test['ubicacion_vial'] = df2test['Descripción'].apply(get_ubicacion_vial)
df2test['lugar_predicted'] = pd.Series(
    list(map(multilabel.predict, desc_vec_dir)))
df2test['lugar_predicted'] = df2test['lugar_predicted'].apply(
    lambda x: dot_prod(lugar, x[0]))

df2test['responsabilidad_pred'] = df2test['responsabilidad_pred'].apply(
    lambda x: x[0])
df2test['lugar_predicted'] = df2test['lugar_predicted'].apply(aux_fun)
df2test[df2test.columns[1:]]

df2test.to_csv('dataset/dataset_final_para_validar_20210506.csv')


# Descomentar esto para una ejecución por consola

# while 1:
#     texto = input('Texto de prueba: ')
#     texto = ' '.join([sbs.stem(word) for word in texto.split()])
#     print('Responsable: ',mlp.predict(vectorizer.transform([texto])))
#     print('Responsable: ',svm.predict(vectorizer.transform([texto])))
#     if texto == '':
#         sys.exit()

# Descomentar esto para generar un csv con las predicciones

# import pandas as pd
# df = pd.read_csv('../dataset/dataset_final_no_dup.csv')
# df['Descripción'] = df['Descripción'].apply(lambda x: ' '.join([sbs.stem(word) for word in x.split()]))
# # df['predicted_mlp'] = df['Descripción'].apply(lambda x: mlp.predict(vectorizer.transform([x]))[0])
# df['svc_predicted_delantera'] = df['Descripción'].apply(lambda x: svc_delantera.predict(vectorizer.transform([x]))[0])
# df['svc_predicted_trasera'] = df['Descripción'].apply(lambda x: svc_trasera.predict(vectorizer.transform([x]))[0])
# from sklearn.metrics import f1_score
# # print('mlp',f1_score(df['responsabilidad'],df['predicted']))
# # df['predicted_svm'] = df['Descripción'].apply(lambda x: svm.predict(vectorizer.transform([x]))[0])
# # print('svm',f1_score(df['responsabilidad'],df['predicted_svm']))
# df.to_csv('dataset_final_predicted_delantera-trasera_svm.csv',index=False)


# Este es el bloque que genera la app web /../


# st.title('Predictor')
# st.markdown('### Describa el siniestro')
# text = st.text_area('', height=50)
# #text = 'choque con mi parte delantera'

# if st.button('Predecir'):
#     text = ' '.join([sbs.stem(word) for word in text.split()])
#     text = vectorizer.transform([text])
#     mlp_pred = mlp.predict(text)
#     svm_pred = svm.predict(text)
# #    delantera_pred = delantera.predict(text)
#  #   trasera_pred = trasera.predict(text)
#     mlp_delantera_pred = mlp_delantera.predict(text)
#     mlp_trasera_pred = mlp_trasera.predict(text)
#     svc_delantera_pred = svc_delantera.predict(text)
#     svc_trasera_pred = svc_trasera.predict(text)
#     if mlp_pred[0] == svm_pred[0] == 1:
#         st.markdown('#### Responsabilidad comprometida')
#     elif mlp_pred[0] == svm_pred[0] == 0:
#         st.markdown('#### Responsabilidad no comprometida')
#     else:
#         st.write('Asesoramiento necesario')

# if delantera_pred[0] == 1:
#     st.markdown('### Colisión con parte delantera')
# elif trasera_pred[0] == 1:
#     st.markdown('### Colisión con parte trasera')
# else:
#     st.markdown('#### Asesoramiento necesario')
