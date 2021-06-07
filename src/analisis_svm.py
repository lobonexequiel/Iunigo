import pickle
import pandas as pd
import matplotlib.pyplot as plt

"""
#Instacio el TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words={'spanish'})
#Entreno y transformo en TRAIN
reviews_train = tfidf.fit_transform(X_train)
reviews_test = tfidf.transform(X_test)
all_words = tfidf.get_feature_names()
# Vizualizamos las 50 palabras mas usadas
print("50 palabras mas usadas: ",all_words[0:50])
17:39
coeff = list(svc.coef_[0])
labels = list(all_words)

"""

with open('models/svm_responsabilidad_20210604.pickle', 'rb') as f:
    svm_model = pickle.load(f)
with open('models/vectorizer_20210504.pickle', 'rb') as v:
    vectorizer = pickle.load(v)

coeff = list(svm_model.coef_[0])
labels = list(vectorizer.get_feature_names())
features = pd.DataFrame()
features['Features'] = labels
features['Importance'] = coeff
features = features.reset_index(drop=True)
features_sort = features.sort_values(by=['Importance'], ascending=True)
features_sort = features_sort.reset_index(drop=True)
data_plot = pd.concat([features_sort.head(50), features_sort.tail(50)])
data_plot.sort_values(by=['Importance'], ascending=False, inplace=True)
data_plot['positive'] = data_plot['Importance'] > 0
data_plot.set_index('Features', inplace=True)
data_plot.Importance.plot(
    kind='bar', figsize=(10, 8),
    color=data_plot.positive.map({True: 'orange', False: 'purple'}),
    fontsize=12)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importancia', rotation=90, fontsize=14)
plt.title('Importancia de los features', fontsize=16)
# plt.savefig('svm_lineal_analisis50.png')
plt.show()
