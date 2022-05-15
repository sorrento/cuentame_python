import pandas as pd

from u_io import lista_files_recursiva, fecha_mod, get_filename
from u_textminig import get_candidatos_nombres_all


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
    df = get_candidatos_nombres_all(texto)
    return pick(df, 10, 2)


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