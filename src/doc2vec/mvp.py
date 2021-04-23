import pickle
from nltk.stem import SnowballStemmer 
sbs = SnowballStemmer('spanish')
with open('mlp_1.pickle','rb') as f:
    mlp = pickle.load(f)
with open('vectorizer1.pickle','rb') as f:
    vectorizer = pickle.load(f)
with open('SVM.pickle','rb') as var:
    svm = pickle.load(var)

texto = input('Texto de prueba: ')
texto = ' '.join([sbs.stem(word) for word in texto.split()])
print('Responsable: ',mlp.predict(vectorizer.transform([texto])))
print('Responsable: ',svm.predict(vectorizer.transform([texto])))
import pandas as pd 
df = pd.read_csv('../../dataset/dataset_final.csv')
df['Descripción'] = df['Descripción'].apply(lambda x: ' '.join([sbs.stem(word) for word in x.split()]))
df['predicted_mlp'] = df['Descripción'].apply(lambda x: mlp.predict(vectorizer.transform([x]))[0])
from sklearn.metrics import f1_score
print('mlp',f1_score(df['responsabilidad'],df['predicted']))
df['predicted_svm'] = df['Descripción'].apply(lambda x: svm.predict(vectorizer.transform([x]))[0])
print('svm',f1_score(df['responsabilidad'],df['predicted_svm']))

df.to_csv('dataset_final_predicted_svm-mlp.csv',index=False)
