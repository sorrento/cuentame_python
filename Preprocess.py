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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # todo: 
# - coger libros random del drive

# %load_ext autoreload
# %autoreload 2
import pandas as pd
from u_base import save_df, save_json, read_json
from utils import get_fakes, get_frecuencia_words, fichero_para_mathematica, agrega_a_dicc, quita_numeros, get_books, \
    get_word_matrix, cabeza_y_cola, corta, crea_capsulas

PATH_CALIBRE = 'c:/Users/milen/Biblioteca de calibre/'

doc_list, files = get_books(PATH_CALIBRE)
vector_matrix, vocab = get_word_matrix(doc_list)
dic_fake, di_counts = get_fakes(doc_list, files, vector_matrix, vocab)

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

# # Fichero para Mathemtica

df_mat, filename_mat = fichero_para_mathematica(dic_fake)

# ## Get partes

i = 6
file = files[i]
texto = doc_list[i]

partes, df = cabeza_y_cola(texto, 30)

ini = 25
fin = 2946
partes, df = corta(partes, df, ini, fin)

d = crea_capsulas(partes, df)

d[1]

i = 17
print(d[i])
print(len(' '.join(d[i]['texto'])))

' '.join(d[i]['texto'])

# # Conexión

import pymongo

# +

jj = {
    'mdb_passw': 'spidey',
    'mdb_usr':   "mhalat"
}
save_json(jj, 'data/config.json')
# -

conf=read_json('data/config.json')

mdb_passw = conf['mdb_passw']
mdb_usr = conf['mdb_usr']

def get_db(mdb_usr, mdb_passw):
    if mdb_usr =='xxx':
        print('debe configurar las credenciales de mongodb en fichero data/config.json.'
              'esta información se encuentra en el panel de back4app por ejemplo')
    cs = "mongodb+srv://" + mdb_usr + ":" + mdb_passw + "@cuentame.2tlxj.mongodb.net/"
    client = pymongo.MongoClient(cs)
    db = client.get_database('cuentame')
    print('**test', db.list_collection_names())
    return db


db = get_db(mdb_passw)


def get_colls(db):
    col_libros_sum = db.get_collection('librosSum')
    col_libros = db.get_collection('libros')
    print('test: nlibros sum (count)', col_libros_sum.count_documents({}))
    print('test libros, example:', col_libros_sum.find_one())

    return col_libros, col_libros_sum


c_lib, c_lib_sum = get_colls(db)

j2 = {'libroId': 999, 'nCapitulos': 999, 'title': 'fake', 'author': 'fake', 'fakeTitle': 'fake', 'fakeAuthor': 'fake',
      'idioma':  'es'}
j3 = {'libroId': 888, 'nCapitulos': 999, 'title': 'fake', 'author': 'fake', 'fakeTitle': 'fake', 'fakeAuthor': 'fake',
      'idioma':  'es'}

dics = [j2, j3]

dics

res = col_libros_sum.insert_many(dics)
