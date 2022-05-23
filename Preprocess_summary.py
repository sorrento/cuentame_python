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

# - recortamos la imagen de acuerdo a dónde no hay texto
#
# # todo:
# - insertamos el registro del libro usando parse e incluyendo la imagen

# %load_ext autoreload
# %autoreload 2

from PIL import Image
from ipywidgets import fixed, interactive
from utils import get_books, crop, get_image_path

PATH_CALIBRE = 'c:/Users/milen/Biblioteca de calibre/'

doc_list, files = get_books(PATH_CALIBRE)

images = [get_image_path(x) for x in files]


im = Image.open(images[2])
im.reduce(4)

u = interactive(crop, f=(0.1, 1,0.05), sx=(1, 200), sy=(1, 500), img=fixed(im))
u

im_r = u.result.resize((150,150))

im_r
