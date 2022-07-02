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
import pandas as pd

from mongo import get_db, get_colls
from u_base import save_df, json_save, json_read
from utils import get_fakes, get_frecuencia_words, fichero_para_mathematica, agrega_a_dicc, quita_numeros, get_books, \
    cabeza_y_cola, corta, crea_capsulas, rompe_parrafo
from u_textmining import get_word_matrix

PATH_CALIBRE = 'c:/Users/milen/Biblioteca de calibre/'

doc_list, files = get_books(PATH_CALIBRE)
vector_matrix, vocab = get_word_matrix(doc_list)
dic_fake, di_counts = get_fakes(doc_list, files, vector_matrix, vocab)

lang = "ES"

# ## para summary:

dic_fake

# +
# Fichero para Mathemtica
# df_mat, filename_mat = fichero_para_mathematica(dic_fake)
# -

# ## Get partes

i_book = 1
file = files[i_book]
texto = doc_list[i_book]

texto

partes, df = cabeza_y_cola(texto, 30)

# todo guardar esto en json, para no hacerlo de nuevo
ini = 13
fin = 386
partes, df = corta(partes, df, ini, fin)

df

la = partes[0]
la


capsu = rompe_parrafo(la)

capsu

d = crea_capsulas(partes, df, lmin=300, lmax=999)

d[1]

d[2]

[len(' '.join(d[x]['texto'])) for x in d]

len(d)

id_free = 999

dic_fake[i_book]["nCapitulos"] = len(d)
dic_fake[i_book]["min"] = ini
dic_fake[i_book]["max"] = fin
dic_fake[i_book]["idioma"] = lang
dic_fake[i_book]["libroId"] = id_free

dic_fake[i_book]

d_summaries={}

j = 'data/summary_ex.json'

d_summaries = json_read(j)

d_summaries[dic_fake[i_book]['title']]=dic_fake[i_book]

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
