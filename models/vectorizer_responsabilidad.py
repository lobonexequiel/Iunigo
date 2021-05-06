from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import spacy

from nltk.stem import SnowballStemmer
sbs = SnowballStemmer('spanish')


VECTORIZER_NAME = 'vectorizador'
RESP_MODEL_NAME = 'svm_responsabilidad'


nlp = spacy.load("es_core_news_sm")
stops = nltk.corpus.stopwords.words('spanish')

# le quitamos los mi y su para el vectorizador.pickle pero no se si para el
# vectorizador1.pickle también se los habíamos quitado

stops = [i for i in stops if i != 'mi']
stops = [j for j in stops if j != 'su']
"""
Descomentar las posibilidades o cambiar el nombre de los archivos a cargar
"""

# df = pd.read_excel('../../dataset/denuncias_iunigo_20210330.xlsx')

df = pd.read_csv('../dataset/dataset_final_20210504.csv', encoding='latin-1')
print(df)
to_del = []

"""
ESTO SE TIENE QUE PODER HACER DE UNA MEJOR MANERA

"""
for i, v in enumerate(df):
    if 'granizo' in df.loc[i, 'descripcion'].lower() or 'cleas' in df.loc[i, 'descripcion'].lower():
        to_del.append(i)
df = df.drop(df.index[to_del])
# df.dropna(inplace=True)
print('Cleanning..')

desc = df['descripcion'].apply(lambda x: ' '.join(
    [sbs.stem(word) for word in x.split()]))
print('Vectorizing..')


vectorizer = TfidfVectorizer(stop_words=stops, ngram_range=(1, 5), min_df=5)

# el vectorizadorse entrena con el texto stemmizado.
vectorizer.fit(desc)

# Se guarda el vectorizador

# with open(VECTORIZER_NAME + '.pickle', 'wb') as f:
#     pickle.dump(vectorizer, f)

# desc_vec son las descripciones vectorizadas
desc_vec = vectorizer.transform(desc)

# la responsabilidad puede ser un vector binario o no
resp_vec = df['responsabilidad']

x_train, x_test, y_train, y_test = train_test_split(
    desc_vec, resp_vec, test_size=0.25)
print(x_train.shape)

print('Training..')
svm_clf = svm.SVC(C=1.5)
svm_clf.fit(x_train, y_train)

# Descomentar para guardar el modelo entrenado

# with open(RESP_MODEL_NAME + '.pickle', 'wb') as f:
#     pickle.dump(svm_clf, f)

y_pred = svm_clf.predict(x_test)

print("SVM")
print('F1 score: ', f1_score(y_test, y_pred))
print('Accuaracy score:', accuracy_score(y_test, y_pred))
