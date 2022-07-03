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

# # Audiobook 
# - Crea audiobooks (mp3) con sintetizador
#
# **to do**
# - verificar si es más rápido leer párraos cortos, sólo los puntos seguidos en el sintetizador

# %load_ext autoreload
# %autoreload 2

from u_base import json_read, json_save, make_folder
from utils import crea_capsulas_max, get_parrafos, get_final_parrfs, speakers_test, get_df_capitulos, \
    get_dic_capitulos, update_di_capi, procesa_capitulo, get_book_datas
from u_textmining import palabras_representativas, get_candidatos_nombres_all

LIM = 950  # largo de las cápsulas, límite de lo que puede leer el sinte

# ## 1. Selección del libro
# Tiene que ser un libro ya procesado, así no tengo que cortar la cabeza y cola desde aquí

txt, im, titulo, d = get_book_datas('nder')

df = get_parrafos(titulo)
df

# ## 2. Creación de cápsulas

final, partes = get_final_parrfs(df, LIM)
final

max(final.len.to_list())  # todo, puede que haya alguno que sea grande y no tenga punto. Cor

final[final.len > LIM]

final[final.len > LIM].parte.iloc[0]

d = crea_capsulas_max(partes, final, lmax=LIM, verbose=False)
caps = ['.\n'.join(d[x]['texto']) for x in d]  # todo probar si sintetizador lee punto aparte

caps[12]  # las cápsulas son las que puede leer de una sola vez

df_caps = get_df_capitulos(caps)
df_caps

di_caps = get_dic_capitulos(df_caps)

# ## 2.1 Descripción de cada capítulo

# +
#depurar los nombres que salen, modificando el regex de split()
# df_names, d_all = get_candidatos_nombres_all(txt)
# list(df_names.index)
# -

capitulos = ['\n '.join(di_caps[cap]['capsulas']) for cap in di_caps]



capitulos_titles = palabras_representativas(capitulos,l_exclude=d['names'])

capitulos_titles

di_caps  # capitulos

d_summaries[titulo]

update_di_capi(di_caps, capitulos_titles, d, titulo)

di_caps[1]

path_book = make_folder('data_out/' + titulo + '/')

json_save(di_caps, path_book + 'content.json')

# ## 2. AUDIO

# ### 2.1 Init

# Atentos a si hay un modelo más moderno que `v3_es`

# +
import torch
from omegaconf import OmegaConf

torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                               'latest_silero_models.yml',
                               progress=False)

models = OmegaConf.load('latest_silero_models.yml')
available_languages = list(models.tts_models.keys())

for lang in available_languages:
    modeli = list(models.tts_models.get(lang).keys())
print(f'Available models for {lang}: {modeli}')

# +
# configuración
language = 'es'
model_id = 'v3_es'

sample_rate = 48000
put_accent = True
put_yo = True

# +
# cargamos el modelo
device = torch.device('cpu')  # or cuda, pero no me funciona

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
model.to(device)  # gpu or cpu
# -

# Atentos a si **aparecen nuevas voces**

speakers_test(model)

# # 3. Creación de mp3 de cada capítulo

j = json_read('data_out/json/' + titulo + '.json', keys_as_integer=True)

procesa_capitulo(j, i_cap=1)
