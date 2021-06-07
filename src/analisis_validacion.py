import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re


def clean_lugar(lugar):
    """
    Symbol’s value as variable is void: text
    """
    patterns = ['delant\w+', 'trase\w+', 'izq\w+', 'derech\w+']
    clean_patterns = ['delantera', 'trasera', 'izquierda', 'derecha']
    for j in range(len(patterns)):
        lugar = re.sub(patterns[j], clean_patterns[j], lugar)
    aux = [i for i in lugar.split() if i in clean_patterns]
    if len(aux):
        return ' '.join(aux)
    return 'no especificado'


df = pd.read_excel(
    'dataset/dataset_validado_20210510.xlsx').drop(columns='Unnamed: 0')


li_pred = df['lugar_predicted']
li = df['lugar_impacto'].apply(clean_lugar)

partes = ['delantera', 'trasera', 'izquierda', 'derecha', 'no especificado']
avg_pres = []
for parte in partes:
    aux, tp, tn, fp, fn = [], [], [], [], []
    for i in range(len(li)):
        if parte in li[i] and parte in li_pred[i]:
            aux.append(1)
            tp.append(1)
        elif parte not in li[i] and parte not in li_pred[i]:
            aux.append(1)
            tn.append(1)
        else:
            aux.append(0)
            if parte in li[i] and parte not in li_pred[i]:
                fn.append(1)
            else:
                fp.append(1)
    avg_pres.append(sum(aux)/len(aux))
    print('precisión de la parte', parte, avg_pres[-1])
    print('f1_score de la parte', parte, sum(
        tp)/(sum(tp) + 0.5*(sum(fn)+sum(fp))))
    print()
np.mean(avg_pres)
delantera = sum([1 for i in li if 'delantera' in i])
trasera = sum([1 for i in li if 'trasera' in i])
izquierda = sum([1 for i in li if 'izquierda' in i])
derecha = sum([1 for i in li if 'derecha' in i])
no_especif = sum([1 for i in li if 'no especificado' in i])
avg_pres = [
    avg_pres[0]*delantera,
    avg_pres[1]*trasera,
    avg_pres[2]*izquierda,
    avg_pres[3]*derecha,
    avg_pres[4]*no_especif
]
avg_pres_w = sum(avg_pres)/(delantera+trasera+izquierda+derecha+no_especif)

# Matrix of precision #########################################################

li = li.apply(lambda x: ' '.join(list(set(x.split()))).strip())
li = li.replace('especificado no','no especificado')
li_pred = li_pred.apply(str.strip)
li_dict = dict(li.value_counts())
li_pred_dict = dict(li.value_counts())

comb = [parte1+' '+parte2 if parte1 !=
        parte2 else parte1 for parte1 in partes for parte2 in partes]
mat = {c: [] for c in comb}

for c in comb:
    for l in range(len(li)):
        if c == li[l] == li_pred[l]:
            mat[c].append(1)
        elif c == li[l] and c != li_pred[l]:
            mat[c].append(0)
                
mat = {k: sum(mat[k])/(len(mat[k])) if len(mat[k]) else 0 for k in mat.keys()}
#mat = {k: sum(mat[k]) if len(mat[k]) else 0 for k in mat.keys()}
mat_final = []
row = []
for i, v in enumerate(mat.values()):
    i += 1
    row.append(v)
    if not i % 5:
        mat_final.append(row)
        row = []

mat_final = np.array(mat_final)

for i in range(len(mat_final)):
    for j in range(len(mat_final)):
        mat_final[j, i] = mat_final[i, j]

sns.heatmap(mat_final,annot=True,xticklabels=partes,yticklabels=partes,cbar_kws={"orientation": "horizontal"})

plt.show()

aux = [1 if 'delantera'==li[i]==li_pred[i] else 0 for i in range(len(li))]
sum(aux)/len(aux)
