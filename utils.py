import pandas as pd

from u_base import get_now_format, inicia, tardado
from u_io import lista_files_recursiva, fecha_mod, get_filename, lee_txt
from u_textminig import get_candidatos_nombres_all, tf_idf_preprocessing


def seleccion_txt(path):
    import pandas as pd
    lista = lista_files_recursiva(path, 'txt')
    fechas = [fecha_mod(x) for x in lista]
    maxi = max(fechas)
    files = [x for x in lista if fecha_mod(x) == maxi]

    print('** La ultima fecha de ficheros es: ', maxi)
    print(pd.DataFrame(get_filename(x) for x in files))

    return files


def get_fake_authors(texto):
    """
devuelve un nombre fake, hecho de la concatenación de dos nombres propios que aparecen en el texto
    :param texto:
    :return: también un df con la lista de los nombres propios y un diccionario con el conteo de todas las palabras
    """
    df_names, d_all = get_candidatos_nombres_all(texto)
    return pick(df_names, 10, 2), df_names, d_all


def get_fake_title(vector_matrix, vocab, i, l_authors=None):
    """
genera un título de 3 palabras con las palabras más representivas del texto
    :param vector_matrix:
    :param vocab:
    :param i:
    :param l_authors:
    :return:
    """
    # df con palabras del libro i y su puntuación
    ej = pd.melt(pd.DataFrame(vector_matrix[i, :].todense(), columns=vocab))
    ejj = ej.sort_values('value', ascending=False).set_index('variable')  # preparamos para la funcion pick

    # quitamos las palabras que se usaron como fake author
    if l_authors is None:
        ejj2 = ejj
    else:
        ejj2 = ejj[~ejj.index.isin([x.lower() for x in l_authors])]

    return pick(ejj2, 15, 3, 'value')


def pick(df, top, n, var_peso='N'):
    """
de un df, coge las top rows y selecciona n índices de acuerdo al peso dado por la columna var_peso
    :param df:
    :param top:
    :param n:
    :param var_peso:
    :return:
    """
    import numpy as np

    df = df.head(top)
    noms = [x.capitalize() for x in df.index]
    #     random.choices(noms, weights=nombres.N,k=3) no puedo hacer sin reemplazo
    pesos = df[var_peso] / sum(df[var_peso])
    l = list(np.random.choice(noms, n, False, pesos))

    return l


def get_book_data(path):
    import re
    partes = re.split(r'[/\\]', path)

    autor = partes[-3]
    ti = re.sub(r' \(\d+\)$', '', partes[-2])

    return {'author': autor, 'title': ti}


def get_book_summary(i, files, doc_list, vector_matrix, vocab):
    """
obtiene los autores fake y reales del libro (y título), ademoas de un diccionario con el conteo de las palabras
    :param i:
    :param files:
    :param doc_list:
    :param vector_matrix:
    :param vocab:
    :return:
    """
    file = files[i]
    texto = doc_list[i]

    di = get_book_data(file)

    # print('\n\n*******', get_filename(file))
    l_authors, df_names, d_count = get_fake_authors(texto)
    fake_authors = ' '.join(l_authors)
    nombres = list(df_names.head(20).index)

    l_title = get_fake_title(vector_matrix, vocab, i, nombres)
    fake_title = ' '.join(l_title)

    di['fake_author'] = fake_authors
    di['fake_title'] = fake_title
    di['path'] = file
    di['listo'] = False
    di['i'] = i

    return di, d_count


def get_fakes(path):
    """
genera un diccioario con el titulo, autor, titulo fake y autor fake para los libros de la biblioteca Calibre que se
han trnasformado en txt en el día más reciente
    :param path: ruta de la biblioteca calibre
    :return:
    """
    t = inicia('Get fakes')
    files = seleccion_txt(path)

    doc_list = [lee_txt(x) for x in files]

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

    di2 = {}
    di_counts = {}
    for i in range(len(files)):
        di, di_count = get_book_summary(i, files, doc_list, vector_matrix, vocab)
        # print(di)
        di2[i] = di
        di_counts[i] = di_count

    tardado(t)

    return di2, di_counts


def get_frecuencia_words(di_counts):
    """
genera un df con conteo de palabras de todos los libros del bunch
    :param di_counts:
    :return:
    """
    import pandas as pd
    import numpy as np
    from functools import reduce

    l = list()
    for i in di_counts.keys():
        df = pd.DataFrame.from_dict(di_counts[i], orient='index').rename(columns={0: i})
        l.append(df)

    final_df = reduce(lambda left, right: left.join(right), l)
    final_df = final_df.replace(np.nan, 0)

    final_df['palabra'] = [x.lower() for x in final_df.index]
    a = pd.melt(final_df, id_vars='palabra', value_name='count')

    conteo = a.groupby('palabra').sum().sort_values('count', ascending=False)

    return conteo


def fichero_para_mathematica(dic_fake):
    """
genera el fichero que necesita mathematica
    :param dic_fake:
    :return:
    """
    oo = {dic_fake[k]['title']: {'titulo': dic_fake[k]['title'],
                                 'author': dic_fake[k]['author'],
                                 'path':   dic_fake[k]['path']} for k in dic_fake}
    filename = get_now_format() + '_' + str(len(dic_fake)) + '.csv'

    return pd.DataFrame.from_dict(oo, orient='index'), filename
