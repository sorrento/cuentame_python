import time


def make_folder(path):
    import os
    try:
        if not os.path.isdir(path):
            print('Creando directorio ', path)
            os.mkdir(path)
        else:
            print('Ya existe: {}'.format(path))
    except OSError:
        print('Ha fallado la creación de la carpeta %s' % path)


def inicia(texto):
    now = time.time()
    print('** Iniciando: {}'.format(texto))

    return [now, texto]


def tardado(lista: list):
    start = lista[0]
    texto = lista[1]
    elapsed = (time.time() - start)
    strftime = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print('** Finalizado {}.  Ha tardado {}'.format(texto, strftime))
    return elapsed


def save_json(dic, path, datos_desc=''):
    """

    :param dic:
    :param path:
    :param datos_desc: sólo para mostrar en un print
    """
    import json
    print('** Guardado los datos ' + datos_desc + ' en {}'.format(path))
    with open(path, 'w', encoding="utf-8") as outfile:
        json.dump(dic, outfile, ensure_ascii=False)


def read_json(json_file):
    import json
    with open(json_file, encoding="utf-8") as in_file:
        data = json.load(in_file)
    return data


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
    :param lista:
    :return:
    """
    from itertools import chain

    # los que no están en anidados los metemos en lista, sino no funciona la iteración
    lista = [[x] if (type(x) != list) else x for x in lista]
    flat_list = list(chain(*lista))

    return sorted(list(set(flat_list)))
