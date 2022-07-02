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

# - recortamos la imagen de acuerdo a dónde no hay texto
#
# # todo:
# - insertamos el registro del libro usando parse e incluyendo la imagen

# %load_ext autoreload
# %autoreload 2

from PIL import Image
from ipywidgets import fixed, interactive
from u_images import crop
from utils import get_books, get_image_path, upload_lib_summary
from u_io import get_filename
from u_base import json_read

PATH_CALIBRE = 'c:/Users/milen/Biblioteca de calibre/'

doc_list, files = get_books(PATH_CALIBRE)

images = [get_image_path(x) for x in files]

i = 1

titulo = get_filename(files[i], True).split(' - ')[0]

im = Image.open(images[i])
if im.size[0] > 700:
    im = im.reduce(2)
im.reduce(4)

u = interactive(crop, f=(0.1, 1, 0.05), sx=(1, int(im.size[0] * .5)), sy=(1, int(im.size[1] * .5)), img=fixed(im))
u

# +
si = u.result.size[0]
a = 200
b = min(si, 2 * a)
im_low = u.result.resize((a, a))
im_hi = u.result.resize((b, b))

im_low
# la insertaremos mano luego
# https://parse-dashboard.back4app.com/apps/a8b7aa27-c240-42d5-9567-d95a43ba4b8f/browser/librosSum
# -

base = 'data_out/images/{}/{}.jpg'
im_low.save(base.format('low', titulo))
im_hi.save(base.format('hi', titulo))

# # Crear el objeto
#
# https://dashboard.back4app.com/apidocs#creating-objects
# https://parse-dashboard.back4app.com/apps/a8b7aa27-c240-42d5-9567-d95a43ba4b8f/browser/librosSum

# +
# vamos a crear el objeto summary para insertarlo en librossum con imagen


# +
# ok, consulta completa funciona
# header = get_headers()
# url = "https://parseapi.back4app.com/classes/librosSum"
# data = requests.get(url, headers=header)
# print(data)
# json_response = data.json()
# print(json_response)
# -

j = json_read('data/summary_ex.json')
j

upload_lib_summary(j)
