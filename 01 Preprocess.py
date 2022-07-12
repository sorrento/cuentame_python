# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # todo: 
# - coger libros random del drive

# %load_ext autoreload
# %autoreload 2

# +
import pandas as pd

from mongo import get_db, get_colls
from ut.base import save_df, json_save, json_read, json_update
from ut.textmining import get_word_matrix
from utils import get_fakes, get_frecuencia_words, agrega_a_dicc, quita_numeros, get_books, \
    cabeza_y_cola, corta, crea_capsulas, rompe_parrafo, get_book_datas, SUMMARIES_JSON,seleccion_txt,txt_read
from ut.io import get_filename
# -

PATH_CALIBRE = 'c:/Users/milen/Biblioteca de calibre/'
lang = "EN"  # >>>

# ## a) Un libro en particular

one_book=True

# i) por el más reciente
last, all_= seleccion_txt(PATH_CALIBRE)
book= get_filename(last[0], True)
book

# ii) alternativamente, por patrón
pat='Huasca'#<<<<<<
book=[get_filename(x, True) for x in all_ if pat in x][0]
book

file=[x for x in all_ if book in x]

# libros de referencia para hacer el tf-idf
date_es=20220703 if lang== 'EN' else 20200504 
files_es, _= seleccion_txt(PATH_CALIBRE, fecha=date_es)
files=file+files_es
doc_list = [txt_read(x) for x in files]

# ## b) De última extracción calibre

PATH_CALIBRE

doc_list, files = get_books(PATH_CALIBRE)
files

# # Continuamos

vector_matrix, vocab, _ = get_word_matrix(doc_list)

dic_fake, di_counts = get_fakes(doc_list, files, vector_matrix, vocab, lang)

if one_book:
    dic_fake={0:dic_fake[0]}

pd.DataFrame.from_dict(dic_fake)

j = {dic_fake[k]['title']: dic_fake[k] for k in dic_fake}
json_update(j, SUMMARIES_JSON)

# ## Get partes

# #### a) por bulk

i_book = 1
file = files[i_book]
texto = doc_list[i_book]

# #### b) Individual (del json)

j = json_read(SUMMARIES_JSON)
titles = sorted(list(j.keys()))
titles

texto, img, titulo, d_summary = get_book_datas('andk')

# #### Continuamos

partes, df = cabeza_y_cola(texto, 30)

# +
fin = 464  # >>>
ini = 1  # >>>


d_summary['min'], d_summary['max'] = ini, fin
# -

d_summary

json_update({d_summary['title']: d_summary}, SUMMARIES_JSON)

# Aquí se puede saltar al 02 si solo se quiere hacer un audiobook

# ## Cortar

partes, df = corta(partes, df, ini, fin)

df

la = partes[0]
la

capsu = rompe_parrafo(la)

capsu

d_summary = crea_capsulas(partes, df, lmin=300, lmax=999)

d_summary[1]

d_summary[2]

[len(' '.join(d_summary[x]['texto'])) for x in d_summary]

len(d_summary)

id_free = 999

dic_fake[i_book]["nCapitulos"] = len(d_summary)
dic_fake[i_book]["min"] = ini
dic_fake[i_book]["max"] = fin
dic_fake[i_book]["idioma"] = lang
dic_fake[i_book]["libroId"] = id_free

dic_fake[i_book]

d_summaries = {}

j = 'data/summary_ex.json'

d_summaries = json_read(j)

d_summaries[dic_fake[i_book]['title']] = dic_fake[i_book]

json_save(d_summaries, j)

dic_fake

# # Conexión

conf = json_read('data/config.json')
db = get_db(conf['mdb_usr'], conf['mdb_passw'])
c_lib, c_lib_sum = get_colls(db)

# ejemplo de inserción
j2 = {'libroId': 999, 'nCapitulos': 999, 'title': 'fake', 'author': 'fake', 'fakeTitle': 'fake', 'fakeAuthor': 'fake',
      'idioma':  'es'}
j3 = {'libroId': 888, 'nCapitulos': 999, 'title': 'fake', 'author': 'fake', 'fakeTitle': 'fake', 'fakeAuthor': 'fake',
      'idioma':  'es'}
dics = [j2, j3]
dics
res = c_lib_sum.insert_many(dics)

# # Update Diccionario

conteo = get_frecuencia_words(di_counts)
dicc_file = agrega_a_dicc(conteo, 'data/diccionario.csv')

dicc_file

dicc_file = quita_numeros(dicc_file)

save_df(dicc_file, 'data', 'diccionario2.csv', True)

aa = pd.read_csv('data/diccionario.csv', sep=';')
aa.shape

aa = pd.read_csv('data/diccionario2.csv_81k_2.csv', sep=';')
aa.shape

dicc_file[dicc_file.n < 4].sample(30)
