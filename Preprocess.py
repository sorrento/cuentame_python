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
from utils import get_fakes

path_calibre = 'c:/Users/milen/Biblioteca de calibre/'

dic_fake = get_fakes(path_calibre)

dic_fake
