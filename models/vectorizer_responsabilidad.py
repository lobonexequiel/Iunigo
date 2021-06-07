from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import nltk
import spacy
import re

from nltk.stem import SnowballStemmer
sbs = SnowballStemmer('spanish')


VECTORIZER_NAME = 'vectorizador_20210607_'
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

df = pd.read_csv('dataset/dataset_final_20210504.csv', encoding='latin-1')
print(df)
to_del = []

"""
ESTO SE TIENE QUE PODER HACER DE UNA MEJOR MANERA

"""
to_del = [
    i
    for i in df.index
    if re.search('cleas',df.loc[i,'descripcion'],re.IGNORECASE)
]
to_del += [
    i
    for i in df.index
    if re.search('grani.*?',df.loc[i,'descripcion'],re.IGNORECASE)
]
df = df.drop(to_del).reset_index(drop=True)
# df.dropna(inplace=True)


print('Cleanning..')

desc = df['descripcion'].apply(lambda x: ' '.join(
    [sbs.stem(word) for word in x.split()]))
print('Vectorizing..')

desc[:10]
vectorizer = TfidfVectorizer(
    stop_words=stops,
    ngram_range=(1, 3),
    min_df=5,
    max_features=5000
)
vectorizer = CountVectorizer(
    stop_words = stops,
    ngram_range = (1,3),
    min_df = 5
)

# el vectorizadorse entrena con el texto stemmizado.
vectorizer.fit(desc)

# Se guarda el vectorizador

with open(VECTORIZER_NAME + '.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)

# desc_vec son las descripciones vectorizadas
desc_vec = vectorizer.transform(desc)

# la responsabilidad puede ser un vector binario o no
resp_vec = df['responsabilidad']

x_train, x_test, y_train, y_test = train_test_split(
    desc_vec, resp_vec, test_size=0.25)
print(x_train.shape)

print('Training..')
svm_clf = svm.SVC(
    C=1.5,
    tol=1e-5)

svm_clf.fit(x_train, y_train)

# Descomentar para guardar el modelo entrenado

# with open(RESP_MODEL_NAME + '_20210604.pickle', 'wb') as f:
#    pickle.dump(svm_clf, f)

y_pred = svm_clf.predict(x_test)

print("SVM")
print('F1 score: ', f1_score(y_test, y_pred))
print('Accuaracy score:', accuracy_score(y_test, y_pred))
