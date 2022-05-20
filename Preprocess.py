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
from u_base import save_df
from utils import get_fakes, get_frecuencia_words, fichero_para_mathematica, agrega_a_dicc, quita_numeros

path_calibre = 'c:/Users/milen/Biblioteca de calibre/'

dic_fake, di_counts = get_fakes(path_calibre)

# # Update Diccionario

conteo = get_frecuencia_words(di_counts)
dicc_file = agrega_a_dicc(conteo, 'data/diccionario.csv')

dicc_file

dicc_file = quita_numeros(dicc_file)

save_df(dicc_file, 'data', 'diccionario2.csv',True)

aa=pd.read_csv('data/diccionario.csv', sep=';')
aa.shape

aa=pd.read_csv('data/diccionario2.csv_81k_2.csv', sep=';')
aa.shape

dicc_file[dicc_file.n<4].sample(30)

# # Fichero para Mathemtica

df_mat, filename_mat = fichero_para_mathematica(dic_fake)

# ## Get partes
