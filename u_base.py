import time


def make_folder(path):
    import os
    try:
        if not os.path.isdir(path):
            print('Creando directorio ', path)
            os.mkdir(path)
        else:
            print('Ya existe: {}'.format(path))
        return path + '/'
    except OSError:
        print('Ha fallado la creación de la carpeta %s' % path)


def inicia(texto):
    ahora = time.time()
    print('\n** Iniciando: {}'.format(texto))

    return [ahora, texto]


def tardado(lista: list):
    start = lista[0]
    texto = lista[1]
    elapsed = (time.time() - start)
    strftime = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print('** Finalizado {}.  Ha tardado {}'.format(texto, strftime))
    return elapsed


def json_save(dic, path, datos_desc=''):
    """

    :param dic:
    :param path:
    :param datos_desc: sólo para mostrar en un print
    """
    import json
    print('** Guardado los datos ' + datos_desc + ' en {}'.format(path))
    with open(path, 'w', encoding="utf-8") as outfile:
        json.dump(dic, outfile, ensure_ascii=False)


def json_read(json_file, keys_as_integer=False):
    import json
    with open(json_file, encoding="utf-8") as in_file:
        data = json.load(in_file)

    if keys_as_integer:
        data = {int(x) if x.isdigit() else x: data[x] for x in data.keys()}

    return data


def json_update(j, path):
    import os
    jj = {}
    if os.path.isfile(path):
        jj = json_read(path)
    jj.update(j)
    json_save(jj, path)


def get_now():
    ct = now()
    # ts = ct.timestamp()
    # print("timestamp:-", ts)

    return str(ct)  # podríamos quedarnos con el objeton (sin str)


def now():
    import datetime
    return datetime.datetime.now()


def get_now_format(f="%Y%m%d"):
    ct = now()
    return ct.strftime(f)


def flatten(lista):
    """
transforma una lista anidada en una lista de componenetes únicos oredenados
OJO: SÓLO SI NO SE REPITEN ELEMENTOS
    :param lista:
    :return:
    """
    from itertools import chain

    # los que no están en anidados los metemos en lista, sino no funciona la iteración
    lista = [[x] if (type(x) != list) else x for x in lista]
    flat_list = list(chain(*lista))

    return sorted(list(set(flat_list)))


def log10p(x):
    """
    equivalente a log1p pero en base 10, que tiene más sentido en dinero
    :param x:
    :return:
    """
    import numpy as np
    return np.log10(x + 1)


def abslog(x):
    """
    función logaritmica que incluye el 0, es espejo en negativos, y es "aproximadamente" base 10
    :param x:
    :return:
    """
    if x < 0:
        y = -log10p(-x)
    else:
        y = log10p(x)
    return y


def save_df(df, path, name, save_index=False, append_size=True):
    if append_size:
        middle = '_' + str(round(df.shape[0] / 1000)) + 'k_' + str(df.shape[1])
    else:
        middle = ''

    filename = path + '/' + name + middle + '.csv'
    print('** Guardando dataset en {}'.format(filename))
    df.to_csv(filename, index=save_index)

    return filename


def win_exe(cmd):
    import os
    from sys import platform
    print("**Executing in Windows shell:" + cmd)
    if platform == 'win32':
        cmd = cmd.replace('/', '\\')
    out = os.popen(cmd).read()
    print('**OUT:{}'.format(out))
    return out


def list_min_pos(lista):
    """
da la (primera) posición del elemento más pequeño
    :param lista:
    :return:
    """
    mi = min(lista)
    return lista.index(mi)