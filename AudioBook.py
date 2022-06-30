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
# - Crea audiobooks (wav o mp3) con sintetizador

# %load_ext autoreload
# %autoreload 2
from u_audio import wav2mp3
from u_base import read_json, make_folder
from utils import crea_capsulas_max, get_parrafos, get_final_parrfs, speakers_test, wav_generator

LIM = 950  # largo de las cápsulas, límite de lo que puede leer el sinte

# ## 1. Selección del libro
# Tiene que ser un libro ya procesado, así no tengo que cortar la cabeza y cola desde aquí

d_summaries = read_json('data/summary_ex.json')
list(set(d_summaries.keys()))

titulo = 'El planeta americano'
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

caps[12]

# ## 3. Creación de wav's base

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

# ## a) generación de los wavs

path = 'data_out/wav/' + titulo
make_folder(path)

# i = 0
for i in range(0, 3):
    wav_generator(caps, 'es_1', i, path, model)

len(caps) * 9

# # Paso a Mp3

# +
# res = wav2mp3(titulo) convierte todos lo de una carpeta

# +
# import ffmpeg
# -

# Es posible hacerlo desde python, pero hay que instalar el ffmpeg y es un poco webiao
from pydub import AudioSegment as qu


# +
# ver si me evito guardar el wav
# -




aa

aa=AudioSegment.from_wav("data_out/wav/test_es_2.wav")
# pa='c:/Users/milen/Desktop/git/cuentame_python/data_out/sample.jpg'
# pa='c:\\Users\\milen\\Desktop\\git\\cuentame_python\\data_out\\sample.jpg'
# pa='data_out\\sample.jpg'
# pa='c:/Users/milen/Downloads/huasca cover.jpg'
pa = 'c:/Users/milen/Downloads/20220629_203445.jpg'
# pa = 'c:/pina.png'

tag = {'title':'micanc', 'artist':'yopisos'}

uu = aa.export("data_out/wav/test_es_t9.mp3", format="mp3", id3v2_version='3',tags=tag, cover=pa).close() # funciona!

# +
# uu = aa.export("data_out/wav/test_es_t11.mp3", format="mp3", id3v2_version='3', cover=pa).close() # NO funciona!

# +
# segun he visto con Picard, las imagenes están, pero el reprodictor 
# no lo muestra porque tiene etiqueta "otro" en vez de "frontal"

# +
# si le agrego que es id3 v2 funciona en ffmpeg
# ffmpeg -i in.mp3 -i sample.jpg -map_metadata 0 -map 0 -map 1 -id3v2_version 3 output.mp3
# -

# # 4 Unión de mp3's
# unimos varios para tener unos más largos como capitulos
#

# simplemente es unir los aufios en un alistay hacer lista.export(...,format='xx')

a1=qu.from_wav('data_out/wav/El planeta americano/0000_es_1.wav')
a2=qu.from_wav('data_out/wav/El planeta americano/0001_es_1.wav')

a1

a2

tag = {'title':'micanc', 'artist':'yopisos'}
pa='c:/Users/milen/Desktop/git/cuentame_python/data_out/sample.jpg'
(a1+ a2).export("data_out/wav/combined.mp3", format="mp3", id3v2_version='3',tags=tag, cover=pa).close() 

# +
# 4.1 nombre de cada capítulo
# -

# ## next
# 1. Unir mp3s (se requiere igual el ffmpeg (ver código en R), así que podemos reeditar la parte de crear mp3s
# 2. generar las palabras carácterísticas de cada capítulo para poner como título
