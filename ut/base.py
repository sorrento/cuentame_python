import time
import pandas as pd
import numpy as np
from datetime import datetime

FORMAT_DATE = "%Y%m%d"
FORMAT_CLASSIC = '%Y-%m-%d'
FORMAT_GRINGO = '%m/%d/%Y'
FORMAT_DATETIME = '%Y-%m-%d %H:%M:%S.%f'
FORMAT_UTC = '%Y-%m-%dT%H:%M:%S.%fZ'
FORMAT_UTC2 = '%Y-%m-%d %H:%M:%S.%f+00:00'


def make_folder(path, verbose=True):
    import os
    try:
        if not os.path.isdir(path):
            print('Creando directorio ', path)
            os.mkdir(path)
        else:
            if verbose: print('Ya existe: {}'.format(path))
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


def json_save(dic, path, datos_desc='', indent=None):
    """

    :param indent:
    :param dic:
    :param path:
    :param datos_desc: sólo para mostrar en un print
    """
    import json
    path2 = path + '.json'
    print('** Guardado los datos ' + datos_desc + ' en {}'.format(path2))
    with open(path2, 'w', encoding="utf-8") as outfile:
        if indent is None:
            json.dump(dic, outfile, ensure_ascii=False)
        else:
            json.dump(dic, outfile, ensure_ascii=False, indent=indent)


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


def get_now(utc=False):
    ct = now(utc)
    # ts = ct.timestamp()
    # print("timestamp:-", ts)

    return str(ct)  # podríamos quedarnos con el objeton (sin str)


def now(utc=False):
    from datetime import timezone, datetime
    if utc:
        tz = timezone.utc
    else:
        tz = None

    return datetime.now(tz)


def get_now_format(f=FORMAT_DATE, utc=False):
    """

    :param utc:
    :param f: ver https://pythonexamples.org/python-datetime-format/
    :return:
    """
    ct = now(utc)
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


def df_save(df, path, name, save_index=False, append_size=True):
    import os
    if append_size:
        middle = '_' + str(round(df.shape[0] / 1000)) + 'k_' + str(df.shape[1])
    else:
        middle = ''

    if not os.path.isdir(path):
        make_folder(path)

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


def in_k(n, dec=0):
    return str(round(n / 1000, dec)) + 'k'


def time_from_str(s, formato, silent=False):
    # https://www.programiz.com/python-programming/datetime/strptime
    try:
        res = datetime.strptime(s, formato)
    except Exception as e:
        if not silent: print(e)
        res = 'no_date'
    return res


def time_to_str(t, formato=FORMAT_DATE):
    return t.strftime(formato)


def seq_len(ini, n, step):
    """
crea una secuencia de n enteros, a distancia step
    :param n:
    :param step:
    :param ini:
    :return:
    """
    end = ini + step * (n + 1)
    return list(np.arange(ini, end, step)[:n])


def nearest(x, lista):
    """
devuelve el número más cercano a x de la lista
    :param x:
    :param lista:
    :return:
    """
    deltas = [abs(s - x) for s in lista]
    pos = list_min_pos(deltas)

    return lista[pos]


def list_min_pos(lista):
    """
da la (primera) posición del elemento más pequeño
    :param lista:
    :return:
    """
    mi = min(lista)
    return lista.index(mi)


def json_from_string(s):
    import json
    return json.loads(s.replace("'", "\""))


def list_freqs(myList):
    frequencyDict = dict()
    visited = set()
    listLength = len(myList)
    for i in range(listLength):
        if myList[i] in visited:
            continue
        else:
            count = 0
            element = myList[i]
            visited.add(myList[i])
            for j in range(listLength - i):
                if myList[j + i] == element:
                    count += 1
            frequencyDict[element] = count
    #     print("Input list is:", myList)
    #     print("Frequency of elements is:")
    #     print(frequencyDict)
    return frequencyDict


def json_update_file(path, dic):
    import os
    existe = os.path.exists(path)
    if existe:
        j = json_read(path)
        j.update(dic)
    else:
        j = dic
    json_save(j, path)


def get_iso_week_from_date(date):
    """Return the ISO week from a date.

    Args:
        date (str): Date

    Returns:
        week (int): ISO week
    """
    date = pd.to_datetime(date)
    return date.isocalendar()


def get_start_date_of_isoweek(year, week, date_format='datetime'):
    """
    Get the start date of the given ISO week using isocalendar()

    Args:
        year (int): Year, i.e. 2022
        week (int): Week, i.e. 34
        date_format (string): Format of the returned date, default is datetime

    Returns:
        datetime.date: Start date of the given ISO week
    """
    from isoweek import Week

    if date_format == 'datetime':
        return Week(year, week).monday()
    else:
        return Week(year, week).monday().strftime(date_format)
