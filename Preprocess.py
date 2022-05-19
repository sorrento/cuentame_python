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
from utils import get_fakes, get_frecuencia_words, fichero_para_mathematica

path_calibre = 'c:/Users/milen/Biblioteca de calibre/'

dic_fake, di_counts = get_fakes(path_calibre)

conteo = get_frecuencia_words(di_counts)
conteo

df_mat, filename_mat= fichero_para_mathematica(dic_fake)

import pandas as pd
dicc_file= pd.read_csv('data/diccionario.csv', sep=';')

dicc_file.head(50) # y el diccionario de inglés? (debería ser por separado)

# +
# TODO QUITAR LOS NÚMEROS Y LAS PALABRAS QUE LLEVAN -
# -

dicc_file['perc']=100*dicc_file['n.total']/dicc_file['n.total'].sum()

dicc_file.head(50)

dicc_file[50:100]

# todo ctualizar fichero de diccionario


# ## Get partes


