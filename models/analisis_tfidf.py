from pprint import pprint
import pandas as pd
import numpy as np
"""
En este programa debemos analizar cuanto cambian los vectores de las palabras que cambiamos.
Por ejemplo: si queremos cambiar conductor por izquierda quiero saber cuanto me va a modificar
el vector de tf-idf
"""
df = pd.read_csv(
    '~/Documentos/LegalHub/Iunigo/dataset/dataset_final_no_dup.csv')

descripcinones = df['Descripción'].apply(str.lower).to_list()[:100]


def tf(word, docs):
    vector = []
    izq_doc_freq = 0
    for doc in docs:
        dic_doc = {word.lower(): 0 for word in doc.split()}
        for w in doc.split():
            if w == word:
                izq_doc_freq += 1
            dic_doc[w] += 1
        izq_freq = sum(list(dic_doc.values()))
        vector.append(izq_doc_freq/(izq_freq+1))
    return np.array(vector)


def idf(word, docs):
    total_df = sum([1 for doc in docs if word in doc])
    n = len(docs)
    return np.log(n/total_df)+1


def tfidf(word, docs):
    """
    Con esta función obtenemos un vector que corresponde al peso tf-idf
    para la palabra word en los documentos docs
    :param word: palabra de la que queremos conocer su tf-idf vector
    :type word: string
    :param docs: total de descripcinones
    :type docs: iterable de strings

    """
    return tf(word, docs)*idf(word, docs)/np.linalg.norm(tf(word, docs)*idf(word, docs))


maxim = min(tfidf('derecho', descripcinones))
for i, v in enumerate(tfidf('derecho', descripcinones)):
    if v == maxim:
        print(descripcinones[i], v)
        break
