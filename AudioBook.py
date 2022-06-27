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
from utils import crea_capsulas_max, get_parrafos, get_final_parrfs

LIM = 950  # largo de las cápsulas, límite de lo que puede leer el sinte

# ## 1. Selección del libro

d_summaries = read_json('data/summary_ex.json')
list(set(d_summaries.keys()))

df = get_parrafos('El planeta americano')
df

# ## 2. Creación de cápsulas

final, partes = get_final_parrfs(df, LIM)
final

max(final.len.to_list())# todo, puede que haya alguno que sea grande y no tenga punto. Cor

final[final.len>LIM]

final[final.len>LIM].parte.iloc[0]

d = crea_capsulas_max(partes, final, lmax=500, verbose=False)
caps = ['.\n'.join(d[x]['texto']) for x in d]  # todo probar si sintetizador lee punto aparte

# ## 3. Creación de wav's base
