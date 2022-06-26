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
from u_io import lee_txt
from utils import corta, crea_capsulas, divide_texto, rompe_parr

j = 'data/summary_ex.json'
d_summaries = read_json(j)
d_summaries

# +
titu = 'El planeta americano'

di = d_summaries[titu]
texto = lee_txt(di['path'])
partes, df = divide_texto(texto, r'\n')
partes, df = corta(partes, df, di['min'], di['max'])
# -

df = df.reset_index().rename(columns={'index': 'i'})
df['ii'] = 0

df

lim = 1000

ies = df[df.len > lim].i.to_list()
df_base = df[~df.i.isin(ies)]

i = ies[0]
df2 = rompe_parr(df, i)

df2

rotos=[rompe_parr(df, i) for i in ies]

final=pd.concat([pd.concat(rotos), df_base]).sort_values(['i', 'ii'])
final

final['i_old']=final.i

final.head()

max(final.len.to_list())

partes=final.parte.to_list()

#redefinimos la i
final['i']=range(len(final))

partes

# # Crea capsulas

d = crea_capsulas(partes, final, lmin=300, lmax=800)
