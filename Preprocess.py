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
from u_io import get_filename, lee_txt
from u_textminig import tf_idf_preprocessing
from utils import seleccion_txt, get_fake_authors, get_fake_title

path_calibre = 'c:/Users/milen/Biblioteca de calibre/'

files = seleccion_txt(path_calibre)

doc_list = [lee_txt(x) for x in files]

# +
params = {
    'tfidf_max_df':          .8,  # proporción de documentos. si lo bajamos quitamos los muy frecuentes
    'tfidf_min_df':          .2,  # % de docs. Si lo subo quito palabras poco frecuentes
    'tfidf_analyzer':        'word',
    'tfidf_stop_words':      True,
    'tfidf_ngram_range_min': 1,
    'tfidf_ngram_range_max': 2,
    'tfidf_strip_accents':   False,
    'tfidf_num_keywords':    5
}

vector_matrix, vocab, doc_freq = tf_idf_preprocessing(doc_list, params)

# print(vocab)
# print(doc_freq)
# print(vector_matrix.todense())

# +
i = 2

print(get_filename(files[i]))
texto = doc_list[i]

l_authors = get_fake_authors(texto)
f_authors = ' '.join(l_authors)

f_authors

l_title = get_fake_title(vector_matrix, vocab, i, l_authors)
# -

f_authors

' '.join(l_title)
