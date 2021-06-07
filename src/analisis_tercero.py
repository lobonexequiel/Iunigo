import pickle
import pandas as pd
from modules import ngrams, charts
from utils import clean
report = pd.read_csv('dataset/report1622044997142.csv',
                     sep=';', encoding='latin-1')
tercero = pd.read_csv(
    'dataset/report1622049194842_relato_tercero.csv', sep=';', encoding='latin-1')

tercero.columns = map(str.lower, tercero.columns)
report.columns = map(str.lower, report.columns)

ter_desc = tercero['Descripción'].apply(
    lambda x: clean.remove_stops(clean.general(x)))
ter_resp = tercero['Motivo de cierre']

data = list(ter_resp.value_counts())
label = list(ter_resp.value_counts().keys())
charts.createPie(data, label)

# 411 reportes del tercero
# 394 nuevos del asegurado

with open('models/vectorizer1.pickle', 'rb') as f:
    vectorizer = pickle.load(f)
with open('models/vectorizer.pickle', 'rb') as f:
    vectorizer_dir = pickle.load(f)

with open('models/svm_responsabilidad.pickle', 'rb') as var:
    svm = pickle.load(var)

with open('models/multilabel.pickle', 'rb') as ml:
    multilabel = pickle.load(ml)

impacto = report['colisión: punto de impacto en asegurado']

ter_desc_vec = vectorizer_dir.transform(ter_desc)
lugar_pred = pd.Series([
    i[0] for i in list(map(multilabel.predict,ter_desc_vec))
])
def dot_prod(string, num):
    aux = []
    for i in range(len(num)):
        if num[i]:
            aux.append(string[i])
    return ' '.join(aux)
lugar = ['trasera', 'delantera', 'izquierda', 'derecha']

lugar_pred = lugar_pred.apply(lambda x: dot_prod(lugar,x))

lugar_pred.value_counts()
df = pd.DataFrame({
    'True':report['colisión: punto de impacto en tercero'],
    'pred':lugar_pred})
df.to_csv('dataset/lugar_tercero_validar_20210607.csv')
