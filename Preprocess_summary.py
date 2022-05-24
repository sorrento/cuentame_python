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
from utils import get_books, crop, get_image_path, upload_lib_summary

PATH_CALIBRE = 'c:/Users/milen/Biblioteca de calibre/'

doc_list, files = get_books(PATH_CALIBRE)

images = [get_image_path(x) for x in files]

im = Image.open(images[3])
if im.size[0] > 700:
    im = im.reduce(2)
im.reduce(4)

u = interactive(crop, f=(0.1, 1, 0.05), sx=(1, int(im.size[0] * .5)), sy=(1, int(im.size[1] * .5)), img=fixed(im))
u

im_r = u.result.resize((200, 200))

im_r
# falta guardar la imagen, la insertaremos mano luego
# https://parse-dashboard.back4app.com/apps/a8b7aa27-c240-42d5-9567-d95a43ba4b8f/browser/librosSum

# # Crear el objeto
#
# https://dashboard.back4app.com/apidocs#creating-objects
# https://parse-dashboard.back4app.com/apps/a8b7aa27-c240-42d5-9567-d95a43ba4b8f/browser/librosSum

# +
# vamos a crear el objeto summary para insertarlo en librossum con imagen
# -


# +
# ok, consulta completa funciona
# header = get_headers()
# url = "https://parseapi.back4app.com/classes/librosSum"
# data = requests.get(url, headers=header)
# print(data)
# json_response = data.json()
# print(json_response)
# -

from u_base import read_json
j = read_json('data/summary_ex.json')
upload_lib_summary(j)

