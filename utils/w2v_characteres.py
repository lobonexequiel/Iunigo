"""
Este modulo debería ser para entrenar un modelo que
resuelva los errores ortográficos a partir de una modificación
del word2vec pero para caracteres
"""
from seaborn import heatmap
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec as w2v
from gensim.models import Phrases
from gensim.test.utils import common_texts

text = []
for i in common_texts:
    aux = []
    for j in ' '.join(i):
        aux.append(j)
    text.append(aux)

model = w2v(sentences=text, window=2, size=300, iter=1000)
common_texts

label = []
for i in text:
    for j in i:
        label.append(j)
label = list(set(label))

text_vec = []
for i in label:
    aux = []
    try:
        text_vec.append(model.wv.get_vector(i))
    except:
        pass


pca = PCA(n_components=10).fit(text_vec)
pca.explained_variance_ratio_.sum()
vec = pca.transform(text_vec)

mat = []
for i in vec:
    aux = []
    for j in vec:
        aux.append(cosine_similarity(i.reshape(1,-1), j.reshape(1,-1)))
    mat.append(aux)
heatmap(mat, xticklabels=label, yticklabels=label)
plt.show()
