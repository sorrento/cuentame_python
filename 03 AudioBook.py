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
# - ejecutar en Collab
# - ejecutar en gpu en local o colab
# - verificar si es más rápido leer párraos cortos, sólo los puntos seguidos en el sintetizador

# %load_ext autoreload
# %autoreload 2

from u_base import json_read, json_save, make_folder, json_update
from utils import crea_capsulas_max, get_parrafos, get_final_parrfs, speakers_test, get_df_capitulos, \
    get_dic_capitulos, update_di_capi, procesa_capitulo, get_book_datas, SUMMARIES_JSON, sample_speaker, test_voices_en, \
    lee, CONTENT_JSON
from u_text import divide_texto_en_dos
from u_textmining import palabras_representativas

LIM = 850  # largo de las cápsulas, límite de lo que puede leer el sinte

# ## 1. Selección del libro
# Tiene que ser un libro ya procesado, así no tengo que cortar la cabeza y cola desde aquí

pat = 'a'
d_summaries = json_read(SUMMARIES_JSON)
# print(d_summaries)
titles = sorted(list(d_summaries.keys()))
titulo = [x for x in titles if pat in x][0]
print(titulo)

d_summary = d_summaries[titulo]

print(d_summary.keys())

# +

texto = txt_read(d_summary['path'])
# -

txt, im, titulo, d_summary = get_book_datas('nder')

# +
# apaño, porque parece que se perdiño en alguna parte
# d_summaries = json_read(SUMMARIES_JSON)
# ini = 91  # >>>
# fin = 3350  # >>>

# d_summaries[titulo]['min'], d_summaries[titulo]['max'] = ini, fin


# json_save(d_summaries, SUMMARIES_JSON)
# -

d_summary.keys()

df = get_parrafos(titulo)
df

# ## 2. Creación de cápsulas

final, partes = get_final_parrfs(df, LIM)
final

max(final.len.to_list())  # todo, puede que haya alguno que sea grande y no tenga punto. Cor

final[final.len > LIM]  # TODO REVISAR POR QUÉ HAY ALGUNOS MÁS LARGOS QUE EL LÍMITE

final[final.len > LIM].parte.iloc[0]

d_capsulas = crea_capsulas_max(partes, final, lmax=LIM, verbose=False)
caps = ['.\n'.join(d_capsulas[x]['texto']) for x in d_capsulas]  # todo probar si sintetizador lee punto aparte

caps[12]  # las cápsulas son las que puede leer de una sola vez

df_capitulos = get_df_capitulos(caps)
df_capitulos

d_capitulos = get_dic_capitulos(df_capitulos)

# ## 2.1 Descripción de cada capítulo

# +
# depurar los nombres que salen, modificando el regex de split()
# df_names, d_all = get_candidatos_nombres_all(txt)
# list(df_names.index)
# -

capitulos = ['\n '.join(d_capitulos[cap]['capsulas']) for cap in d_capitulos]

len(capitulos)

capitulos_titles = palabras_representativas(capitulos, l_exclude=d_summary['names'])
capitulos_titles

update_di_capi(d_capitulos, capitulos_titles, d_summary, titulo)

d_capitulos[1].keys()

path_book = make_folder('data_out/' + titulo + '/')

json_save(d_capitulos, path_book + CONTENT_JSON)

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
language = d_summary['idioma'].lower()
model_id = 'v3_es' if language == 'es' else 'v3_en'

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

if language == 'es':
    speakers_test(model)

# +
# speakers, random voice
if 'speaker' not in d:
    if language == 'es':
        speaker = 'es_1'
    else:
        import random

        best_en = ['en_' + str(i) for i in [33, 50, 61, 75, 94]]
        speaker = random.choice(best_en)

    # update fichero
    d['speaker'] = speaker
    json_update({titulo: d}, SUMMARIES_JSON)

else:
    speaker = d['speaker']

# -


speaker = 'en_94'  # Sophie
d_summary['speaker'] = speaker
json_update({titulo: d_summary}, SUMMARIES_JSON)

test_voices_en(model, best_en)

sample_speaker(model, d_summary)

# # 3. Creación de mp3 de cada capítulo

path_json = 'data_out/{}/{}'.format(titulo, CONTENT_JSON)
d_capitulos = json_read(path_json, keys_as_integer=True)

ini = 3
for i_cap in range(ini, 25 + 1):
    procesa_capitulo(d_capitulos, i_capitulo=i_cap, titulo=titulo, path_book=path_book, model=model,
                     speaker=speaker,
                     debug_mode=False)

# Prueba si un cap falla por longitud
txt = d_capitulos[3]['capsulas'][28]
print(len(txt))
print(txt)
lee(model, txt[:750], speaker)
