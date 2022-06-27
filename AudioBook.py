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
from u_base import read_json
from utils import crea_capsulas_max, get_parrafos, get_final_parrfs, speakers_test, lee, reemplaza_nums
from u_text import numero_a_letras

LIM = 950  # largo de las cápsulas, límite de lo que puede leer el sinte

# ## 1. Selección del libro

d_summaries = read_json('data/summary_ex.json')
list(set(d_summaries.keys()))

df = get_parrafos('El planeta americano')
df

# ## 2. Creación de cápsulas

final, partes = get_final_parrfs(df, LIM)
final

max(final.len.to_list())  # todo, puede que haya alguno que sea grande y no tenga punto. Cor

final[final.len > LIM]

final[final.len > LIM].parte.iloc[0]

d = crea_capsulas_max(partes, final, lmax=500, verbose=False)
caps = ['.\n'.join(d[x]['texto']) for x in d]  # todo probar si sintetizador lee punto aparte

# ## 3. Creación de wav's base

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
# -

device = torch.device('cpu')  # or cuda, pero no me funciona

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
model.to(device)  # gpu or cpu

speakers_test(model)
