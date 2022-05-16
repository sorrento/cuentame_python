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
from u_base import get_now
from utils import get_fakes, get_frecuencia_words, fichero_para_mathematica

path_calibre = 'c:/Users/milen/Biblioteca de calibre/'

dic_fake, di_counts = get_fakes(path_calibre)

conteo = get_frecuencia_words(di_counts)
conteo

df_mat, filename_mat= fichero_para_mathematica(dic_fake)


