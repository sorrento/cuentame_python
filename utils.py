import os
import numpy as np
import pandas as pd
import requests
import json

from IPython.core.display import display
from IPython.lib.display import Audio
from pydub import AudioSegment

from conf.secrets_k import *
from utils_base import get_now_format, inicia, tardado, json_read, make_folder, json_update
from utils_io import lista_files_recursiva, fecha_mod, get_filename, txt_read, txt_write
from utils_plots import plot_hist
from utils_text import divide_texto_en_dos, number_to_text
from utils_textmining import get_candidatos_nombres_all, pick

SAMPLE_EN = 'The monitor lady smiled very nicely and tousled his hair and said, "Andrew, I suppose by now you\'re just absolutely sick of having that horrid monitor. Well, I have good news for you. That monitor is '
SAMPLE_ES = 'Formalmente, desde el Acuerdo Marco "Aurora" de 1953, los centros pertenecientes a la red mundial debían ' \
            '"trabajar en plena colaboración académica y humana, compartiendo los avances tanto en conocimientos ' \
            'fundamentales como en técnicas.'

SUMMARIES_JSON = 'data_med/summaries.json'
CONTENT_JSON = 'capitulos.json'

########### PRINCIPALES ####################


def get_books(path):
    files, _ = seleccion_txt(path)
    doc_list = [txt_read(x) for x in files]
    return doc_list, files


def get_fakes(doc_list, files, vector_matrix, vocab, lang, openAI=False):
    """
genera un diccioario con el titulo, autor, titulo fake y autor fake para los libros de la biblioteca Calibre que se
han trnasformado en txt en el día más reciente
    :param vocab:
    :param vector_matrix:
    :param files:
    :param doc_list:
    :return:
    """
    t = inicia('Get fakes')

    di_fakes = {}
    di_counts = {}
    for i in range(len(files)):
        di_fake, di_count = get_book_summary(i, files, doc_list, vector_matrix, vocab, openAI)
        # print(di_fake)
        di_fake['idioma'] = lang
        di_fakes[i] = di_fake
        di_counts[i] = di_count

    tardado(t)

    return di_fakes, di_counts


def get_book_datas(pat):
    '''
    Devuelve el texto, la imagen, el título y el diccionario de summary, que lo
    :param pat: patrón de búsqueda
    :return: texto, im, titulo, d_summary

    '''
    from PIL import Image
    d_summaries = json_read(SUMMARIES_JSON)

    titles = sorted(list(d_summaries.keys()))
    matches = [x for x in titles if pat in x]
    if len(matches) > 0:
        titulo = matches[0]
    else:
        print('No hay matches en \n{}'.format(titles))
        return None, None, None, None
    print(titulo)
    d_summary = d_summaries[titulo]
    path = d_summary['path']
    texto = txt_read(path)
    image_path = get_image_path(path)
    try:
        im = Image.open(image_path)
    except:
        print('No se pudo abrir la imagen "{}"'.format(image_path))
        im = None

    return texto, im, titulo, d_summary


def cabeza_y_cola(texto, n_row=100):
    """
muestra el principio y el final, para que podamos a ojo ver dónde empieza y termina realmente el libro
    :param texto:
    :param n_row: cuantas filas muestra
    :return:
    """
    from IPython.core.display import display
    partes, df = divide_texto(texto, r'\n')

    pd.set_option('display.width', 10)
    pd.set_option('display.max_colwidth', 150)
    pd.set_option('display.max_rows', 300)

    display(df.tail(n_row))
    display(df.head(n_row))

    return partes, df


def corta(partes, df, ini, fin):
    """
corta el df y la lista con las partes usando los índices encontrados
    :param partes:
    :param df:
    :param ini:
    :param fin:
    :return:
    """
    df = df[(df.index >= ini) & (df.index <= fin)].reset_index(drop=True)
    partes = [partes[x] for x in range(ini, fin + 1)]
    return partes, df


def crea_capsulas(partes, df, lmin=1000, lmax=1500, verbose=True):
    """
iva uniendo las partes hasta juntarlas en capsulas de tamaño entre lmin y lmax. El resultado es un diccionario
que tiene 'ies' (las i que une) y 'texto'. la key es índice de grupo g que parte en 1
    :param partes:
    :param df:
    :param lmin:
    :param lmax:
    :return:
    """
    grupos = []  # lista de a qué grupo pertenece la fila del df
    largos = []  # almacena los largos de las capsulas creadas
    n_acc = 0  # acumulado de la suma de largos en la iteración
    g = 1  # id de grupo
    d = {}  # diccionario final que se entregará

    for i in range(len(df)):
        r = df.iloc[i]
        n_new = r.len

        n_fut = n_acc + n_new
        if verbose:
            print('*****', i)
            print('******* nacc={} nnew={} nfut={}'.format(n_acc, n_new, n_fut))

        q_cortos = n_fut <= lmin
        q_largos = n_fut > lmax

        if q_cortos:
            #         print('  **cortos', n_fut)
            agg(grupos, g, d, i, partes)
            n_acc = n_fut
        elif q_largos:
            print('  ** >> pasamos', n_fut)
            delta_abajo = lmin - n_acc
            delta_arriba = n_fut - lmax
            print('delta abajo: {}, delta arriba {} nacc {}  nnew{} nfut {}'.format(delta_abajo, delta_arriba, n_acc,
                                                                                    n_new, n_fut))
            if delta_abajo < delta_arriba:
                print('>>preferimos quedarnos cortos')
                g, n_acc = agrega(grupos, largos, g, n_acc, i, d, partes)
                n_acc = n_new
            else:
                print('>>> preferimos pasarnos', n_fut)
                g, n_acc = agrega(grupos, largos, g, n_fut, i, d, partes)

        else:
            if verbose:
                print('***caemos dentro:', n_fut)
            g, n_acc = agrega(grupos, largos, g, n_fut, i, d, partes)

    df['capsula'] = grupos
    if verbose:
        plot_hist(largos, 45)

    return d

########### PARSE ####################


def get_all_books_summaries():
    """
    Trae todos los libros de la base de datos de resúmenes haciendo la paginación
    """
    limit = 300  # número de libros a traer cada vez
    skip = 0  # número de libros a saltar
    hay_mas = True

    url = "https://parseapi.back4app.com/classes/librosSum"
    header = get_parse_headers()
    condicion = '{"libroId":{"$exists":true}}'

    def trae(limit, skip):
        params = {'where': condicion, 'limit': limit, 'skip': skip}
        data = requests.get(url, headers=header, params=params)
        # si la respuesta es distinta de 200, hay un error
        if data.status_code != 200:
            raise (f'Error {data.status_code} en la llamada a la API')
        return data.json()['results']

    i = 1
    libros = []
    while hay_mas:
        print(f'iteración {i}')
        data = trae(limit, skip)
        if len(data) == 0:
            hay_mas = False
        else:
            print
            skip += limit
            libros += data
            i += 1
    # params = {'where': condicion, 'limit': 100, 'skip': 200}
    # data = requests.get(url, headers=header, params=params)
    print(f'Hemos terminado. Hay {len(libros)} libros')
    return libros


class Biblioteca:
    def __init__(self) -> None:
        self.data = get_all_books_summaries()

    def get_ids(self):
        return sorted([x.get('libroId') for x in self.data])

    def borra_libro(self, libroId):
        print('No implementado (borra_libro)')

    def get_titulos(self):
        """
        Tuplas id, titulo
        """
        return [(x.get('libroId'), x.get('title')) for x in self.data]

    def get_next_id_available(self):
        if len(self.candidatos_id) == 0:
            print('No hay más ids disponibles')
            return None
        # cogemos el más pequeño de los candidatos y lo borramos de la lista
        id = min(self.candidatos_id)
        self.candidatos_id.remove(id)
        return id

    def save_locally(self):
        import yaml
        with open('data_med/biblio.yml', 'w') as outfile:
            yaml.dump(self.data, outfile, default_flow_style=False)
        # diccionario simplificado {libroId: {title, author}}
        print(f'** Guardado en data_med/biblio.yml')

        d_simple = {v['libroId']: {'title': v['title'], 'author': v['author']} for v in self.data}
        d_simple
        with open('data_med/biblio_simple.yml', 'w') as outfile:
            yaml.dump(d_simple, outfile, default_flow_style=False)
        print(f'** Guardado en data_med/biblio_simple.yml')

        # como csv
        df = pd.DataFrame.from_dict(d_simple, orient='index')
        df = df.sort_index()
        df.to_csv('data_med/biblio_simple.csv')
        print(f'** Guardado en data_med/biblio_simple.csv')


def get_ids_disponibles():
    biblio = Biblioteca()
    ids = biblio.get_ids()
    all_ids = range(1, max(ids)+1)
    huecos = [x for x in all_ids if x not in ids]
    # agregamos los 30 siguientes al mayor
    huecos += range(max(ids)+1, max(ids)+30)
    return huecos


def get_next_id_available():
    ids = get_ids_disponibles()
    print(f'El siguiente id disponible es {min(ids)}')
    return min(ids)


def upload_lib_summary(j):
    url = "https://parseapi.back4app.com/classes/librosSum/"

    j.pop('path')
    j.pop('listo')
    j.pop('i')
    # también names, min, max
    j.pop('names')
    j.pop('min')
    j.pop('max')

    libroId = j['libroId']
    title = j['title']
    print(f'** Subiendo libro {libroId} {title}')
    v = str(j).replace('\'', '\"').encode('utf-8')
    header = get_parse_headers()
    data = requests.post(url, data=v, headers=header)

    print(data.json())


def get_parse_headers():
    return {
        'X-Parse-Application-Id': X_PARSE_APPLICATION_ID,
        'X-Parse-REST-API-Key':   X_PARSE_REST_API_KEY,
        'Content-Type':           'application/json'
    }


def update_status(objectId):
    url = "https://parseapi.back4app.com/classes/WordCorpus/" + objectId
    payload = {'status': True}
    header = get_parse_headers()
    response = requests.put(url, data=json.dumps(payload), headers=header)
    print(response.text)
    return response.status_code


class Back4App:
    def get_sentance(self):
        header = get_parse_headers()
        url = "https://parseapi.back4app.com/classes/WordCorpus?where=%7B%22status%22%3Afalse%7D"
        data = requests.get(url, headers=header)
        print(data)
        json_response = data.json()
        print(json_response)
        results = json_response['results'][0]
        # for i in results['meaning']:
        # print(i)

        sentence = ("சொல் : %s \n பொருள் : %s" %
                    (results['word'], results['meaning']))
        update_status(results['objectId'])
        tags = "\n#தினமொரு #தமிழ்_சொல்"
        return sentence + tags

# buscamos el object id


def get_object_id(libroId):
    import yaml
    with open('data_med/biblio.yml', 'r', encoding='utf-8') as f:
        biblio = yaml.load(f, Loader=yaml.FullLoader)

    # buscamos el objectId del que tiene libroId=id_borrar
    for x in biblio:
        if x['libroId'] == libroId:
            return (x['objectId'])

    print(f'No se encontró libroId={libroId}')


def borrar_de_summary(id_borrar):
    from utils import get_parse_headers, get_object_id
    import requests
    headers = get_parse_headers()
    obj_id = get_object_id(id_borrar)

    url = 'https://parseapi.back4app.com/classes/librosSum/'
    url = url + obj_id
    headers = headers
    r = requests.delete(url, headers=headers)
    print(r.status_code)


def borrar_batch_capsulas(id_borrar):
    # borramos masivamente las capsulas de 'libros'
    # 1. traemos el campo 'objectId' de la tabla 'libros' de los que tienen 'nLibro' = id_borrar
    import requests
    from utils import get_parse_headers
    url = 'https://parseapi.back4app.com/classes/libros/'

    headers = get_parse_headers()
    conditions = '{"nLibro":'+str(id_borrar)+'}'
    params = {'where': conditions}  # , 'keys': 'objectId'}
    r = requests.get(url, headers=headers, params=params)

    results = r.json()['results']
    len(results)
    print(f'Hay {len(results)} capsulas con nLibro={id_borrar}')
    lista = []
    for x in results:
        lista.append({'method': 'DELETE', 'path': '/1/classes/libros/'+x['objectId']})
    # Los datos deben estar bajo la clave 'requests', y su valor debe ser la lista de operaciones.
    data = {'requests': lista}

    url = 'https://parseapi.back4app.com/batch'
    # url= 'https://api.parse.com/1/batch'
    headers = get_parse_headers()
    r = requests.post(url, headers=headers, json=data)
    r.status_code
    if r.status_code == 200:
        print('Borrado masivo de libros exitoso')
    else:
        print('Error en borrado masivo de libros')
        print(r.json())

########### OTRAS ####################


def seleccion_txt(path, fecha=None):
    all_ = lista_files_recursiva(path, 'txt')
    fechas = [fecha_mod(x) for x in all_]
    if fecha is None:
        fecha = max(fechas)
        print('** La ultima fecha de ficheros es: ', fecha)

    files = [x for x in all_ if fecha_mod(x) == fecha]

    print(pd.DataFrame(get_filename(x) for x in files))

    return files, all_


def get_fake_authors(texto):
    """
devuelve un nombre fake, hecho de la concatenación de dos nombres propios que aparecen en el texto
    :param texto:
    :return: también un df con la lista de los nombres propios y un diccionario con el conteo de todas las palabras
    """
    df_names, d_all = get_candidatos_nombres_all(texto)
    return pick(df_names, 10, 2), df_names, d_all


def get_fake_title(vector_matrix, vocab, i, l_authors=None, with_openAI=False):
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

    # quitamos los numeros:
    digits = [x for x in ejj.index if x.isdigit()]
    ejj = ejj[~ejj.index.isin(digits)]

    # quitamos las palabras que se usaron como fake author
    if l_authors is None:
        ejj2 = ejj
    else:
        ejj2 = ejj[~ejj.index.isin([x.lower() for x in l_authors])]

    if with_openAI:
        # convert to dict las primeras 20 filas {indice:value}
        di = ejj2.iloc[:20].to_dict()['value']
        res = [genera_titulo_openAI(di)]
    else:
        res = pick(ejj2, 15, 3, 'value')  # esto es una lista de 3 palabras, el 15

    return res


def get_book_data(path):
    import re
    partes = re.split(r'[/\\]', path)

    autor = partes[-3]
    ti = re.sub(r' \(\d+\)$', '', partes[-2])

    return {'author': autor, 'title': ti}


def get_book_summary(i, files, doc_list, vector_matrix, vocab, openai=False):
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
    nombres = list(df_names.index)

    l_title = get_fake_title(vector_matrix, vocab, i, nombres, with_openAI=openai)
    fake_title = ' '.join(l_title)

    print(f'  {fake_title} ({di["title"]})')

    di["fakeAuthor"] = fake_authors
    di["fakeTitle"] = fake_title
    di["path"] = file
    di["listo"] = False
    di["i"] = i
    di['names'] = nombres

    return di, d_count


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

    col_word = 'word'
    final_df[col_word] = [x.lower() for x in final_df.index]
    a = pd.melt(final_df, id_vars=col_word, value_name='count')

    conteo = a.groupby(col_word).sum().sort_values('count', ascending=False)

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


def agrega_a_dicc(conteo, path='data/diccionario.csv'):
    """
lee el fichero de conteo de palabras y le agrega el nuevo
    :param conteo:
    :param path:
    :return:
    """
    dicc_file = pd.read_csv(path, sep=';', index_col='word')
    # y el diccionario de inglés? (debería ser por separado)
    print('antes', dicc_file.shape)
    res = conteo.join(dicc_file, how='outer').replace({np.nan: 0})
    res['n'] = res['count'] + res['n.total']
    res = res.sort_values('n', ascending=False)
    res['r'] = np.arange(1, 0, -(1 / len(res)))
    print('después:', res.shape)

    return res[['n', 'r']]


def quita_numeros(dicc_file):
    """
quita de un df de conteo de palabras, aquellas que en realidad son números
    :param dicc_file:
    :return:
    """
    print('antes:', dicc_file.shape)
    lista = [x for x in dicc_file.index if not x.isdigit()]
    res = dicc_file[dicc_file.index.isin(lista)]
    print('después:', res.shape)
    return res


def divide_texto(texto, pat):
    """
Separa el texto en cada apaarición del patrón.
    :param texto:
    :param pat:
    :return:
    """
    import re
    partes = [x for x in re.split(pat, texto) if x != '']
    df = pd.DataFrame({'i': range(len(partes)), 'parte': partes}).set_index('i')
    df['len'] = df.parte.map(len)
    return partes, df


def agrega(l, largo, g, n_new, i, d, partes):
    agg(l, g, d, i, partes)
    largo.append(n_new)

    return g + 1, 0


def agg(l, g, d, i, partes):
    # lista de a qué grupo va
    l.append(g)
    #     print('largo l:',len(l))

    # agregamos a diccionario
    if g in d:
        ies = d[g]['ies']
        ies.append(i)
        textos = d[g]['texto']
        textos.append(partes[i])
    else:
        d[g] = {'ies': [i]}
        d[g]['texto'] = [partes[i]]


def crea_capsulas_max(partes, df, lmax=1000, verbose=True):
    """
va uniendo las partes hasta juntarlas en capsulas antes de alcanzar el  lmax. El resultado es un diccionario
que tiene 'ies' (las i que une) y 'texto'. la key es índice de grupo g que parte en 1
    :param partes:
    :param df:
    :param lmax:
    :return:
    """
    grupos = []  # lista de a qué grupo pertenece la fila del df
    largos = []  # almacena los largos de las capsulas creadas
    n_acc = 0  # acumulado de la suma de largos en la iteración
    g = 1  # id de grupo
    d = {}  # diccionario final que se entregará

    for i in range(len(df)):
        r = df.iloc[i]
        n_new = r.len

        n_fut = n_acc + n_new
        if verbose:
            print('*****', i)
            print('******* nacc={} nnew={} nfut={}'.format(n_acc, n_new, n_fut))

        q_largos = n_fut > lmax

        if q_largos:
            if verbose:
                print('  ** >> pasamos', n_fut)
            largos.append(n_new)  # cerramos el anterior
            g = g + 1
            agg(grupos, g, d, i, partes)
            n_acc = n_new
        else:
            agg(grupos, g, d, i, partes)
            n_acc = n_fut

    df['capsula'] = grupos
    if verbose:
        plot_hist(largos, 45)

    if verbose:
        print('** Máximo largo: {}, mínimo: {}'.format(str(max(largos)), str(min(largos))))
    return d


def get_image_path(file):
    oo = file.split('\\')[:-1]
    oo.append('cover.jpg')
    return '/'.join(oo)


def rompe_parrafo(la, lim):
    partes, df = divide_texto(la, r'\. ')
    dd = crea_capsulas_max(partes, df, lmax=lim, verbose=False)
    capsu = ['. '.join(dd[x]['texto']) + '.' for x in dd]
    print('largos:', [len(x) for x in capsu])
    return capsu


def rompe_parr(df, i, lim):
    row = df[df.i == i]
    la = row.parte.iloc[0]
    # print('rompiendo parr ', str(i))
    capsu = rompe_parrafo(la, lim)
    les = [len(x) for x in capsu]
    df2 = pd.DataFrame({'i': i, 'parte': capsu, 'ii': range(len(capsu)), 'len': les})
    return df2


def get_parrafos(titu):
    d_summaries = json_read(SUMMARIES_JSON)

    di = d_summaries[titu]
    texto = txt_read(di['path'])
    partes, df = divide_texto(texto, r'\n')
    partes, df = corta(partes, df, di['min'], di['max'])
    df = df.reset_index().rename(columns={'index': 'i'})
    df['ii'] = 0  # para identificar dentro de un párrafo largo que romperemos

    return df


def get_final_parrfs(df, LIM):
    ies = df[df.len > LIM].i.to_list()
    df_base = df[~df.i.isin(ies)]
    rotos = [rompe_parr(df, i, LIM) for i in ies]
    final = pd.concat([pd.concat(rotos), df_base]).sort_values(['i', 'ii'])
    final['i_old'] = final.i
    final['i'] = range(len(final))
    partes = final.parte.to_list()

    return final, partes


def reemplaza_nums(new_string, lan):
    """
convierte los numero en un texto en strings con los números en palabras
    :param new_string:
    :return:
    """
    import re
    # new_string = 'Rose67lilly78Jasmine228Tulip'
    new_result = re.findall('[0-9]+', new_string)

    if len(new_result) == 0:
        return new_string

    dic = {x: number_to_text(int(x), lan) for x in new_result}
    for key, value in dic.items():
        # Replace key character with value character in string
        new_string = new_string.replace(key, value)
    return new_string

### ############## WAV GENERATOR ####################


def wav_generator(txt, voz, i_cap, path, model, write_txt=True,
                  sample_rate=48000, put_accent=True, put_yo=True, n_caps='?', i_capitulo='?', lan='en', save_mp3=True):

    params = {'voz': voz, 'i_cap': i_cap, 'path': path, 'model': model, 'write_txt': write_txt,
              'sample_rate': sample_rate, 'put_accent': put_accent, 'put_yo': put_yo, 'n_caps': n_caps, 'i_capitulo': i_capitulo, 'lan': lan, 'save_mp3': save_mp3}

    aug_seg = wav_genertor_general(txt, 'SILERO', **params)


def wav_genertor_general(txt, motor, **kwargs):

    t = inicia(' capsula = {}/{}. Capitulo:{}'.format(i_cap, n_caps, i_capitulo))
    print(f'Utilizamos el motor {motor} para leer el texto: {txt[:100]}')

    name = str(i_cap).zfill(4) + '_' + voz
    mp_ = path + name + '.mp3'

    if os.path.isfile(mp_):
        print('*ya existe la parte {}. La saltamos '.format(mp_))
        au_seg = AudioSegment.from_mp3(mp_)
    else:
        if motor == 'SILERO':
            res = wav_generator_silero(txt, **kwargs)
        elif motor == 'GOOGLE':
            res = wav_generator_google(txt, **kwargs)
        elif motor == 'AZURE':
            res = wav_generator_azure(txt, **kwargs)
    if write_txt:
        txt_write(path + '/' + name, txt)

    tardado(t)

    return res


def wav_generator_silero(txt, voz, i_cap, path, model, write_txt=True,
                         sample_rate=48000, put_accent=True, put_yo=True, n_caps='?', i_capitulo='?', lan='en', save_mp3=True):

    try:
        audio = model.apply_tts(text=reemplaza_nums(txt, lan),
                                speaker=voz,
                                sample_rate=sample_rate,
                                put_accent=put_accent,
                                put_yo=put_yo
                                )
        au = Audio(audio, rate=sample_rate)
        au_seg = audio_save(au, name, path, save_mp3=save_mp3)
    except Exception as e:
        # tipicamente ocurre porque es muy largo, así que lo hacemos en dos partes y juntamos
        print('** ERROR: ' + str(e))
        txt1, txt2 = divide_texto_en_dos(txt)
        au1 = wav_generator(txt1, voz, str(i_cap) + '_a', path, model, save_mp3=False)
        au2 = wav_generator(txt1, voz, str(i_cap) + '_b', path, model, save_mp3=False)
        au_seg = au1 + au2

    return au_seg


def audio_save(au, name, path, mp3=True, show=False, tag=None, save_mp3=True):
    print(
        f' in audio_save, params: name: {name}, path: {path}, mp3: {mp3}, show: {show}, tag: {tag}, save_mp3: {save_mp3}')
    tem = 'temp.wav'
    wa = None
    if mp3:
        ext = '.mp3'
    else:
        ext = '.wav'

    no = path + '/' + name + ext
    print(' ** Guardando ', no)

    if mp3:
        with open(tem, 'wb') as f:
            f.write(au.data)

        wa = AudioSegment.from_wav(tem)
        # wa.export(no, format="mp3", tags=tag).close()
        if save_mp3:
            wa.export(no, format="mp3").close()
        os.remove(tem)
    else:
        with open(no, 'wb') as f:
            f.write(au.data)

    if show:
        display(au)

    print(' saliendo de audio_save')

    return wa


def speakers_test(model, put_accent=True, sample_rate=48000, put_yo=True,
                  txt='Formalmente, desde el Acuerdo Marco "Aurora" de 1953, los centros pertenecientes a la red '
                      'mundial debían "trabajar en plena colaboración académica y humana, compartiendo los avances '
                      'tanto en conocimientos fundamentales como en técnicas.',
                      lan='es'):
    print(txt)
    sps = [x for x in model.speakers if x != 'random']
    for sp in sps:
        # print(sp)
        audio = model.apply_tts(text=reemplaza_nums(txt, lan='es'),
                                speaker=sp,
                                sample_rate=sample_rate,
                                put_accent=put_accent,
                                put_yo=put_yo
                                )
        au = Audio(audio, rate=sample_rate)
        print(f'** {sp} **')
        display(au)
        tag = {'title': 'Voice Test: ' + sp, 'artist': sp}
        # audio_save(au, 'test_' + sp, 'data_out/wav/', show=True, tag=tag)


def lee(model,
        txt=SAMPLE_ES, speaker='es_1', sample_rate=48000, put_accent=True, put_yo=True,
        lan='en'):
    audio = model.apply_tts(text=reemplaza_nums(txt, lan),
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo
                            )
    au = Audio(audio, rate=sample_rate)
    # guarda_wav(au, 'te_' + sp)
    return au


def sample_speaker(model, d):
    if d['idioma'] == 'ES':
        txt = SAMPLE_ES[:200]
    else:
        txt = SAMPLE_EN
    return lee(model, txt, speaker=d['speaker'])


def test_voices_en(model, lista=None, d_capitulos=None, n=4, avoid=None, text=None):
    if text is None:
        txt = d_capitulos['1']['capsulas'][0][:230] if (d_capitulos is not None) else SAMPLE_EN
    else:
        txt = text[:230]
    print(txt)

    if lista is None:
        import random
        all_ = model.speakers
        if avoid is None:
            available = all_
        else:
            available = list(set(all_) - set(avoid))
        lista = random.sample(available, k=n)

    for vo in lista:
        print(vo)
        display(lee(model, txt, speaker=vo))

    return lista


def get_mp3_tag(d_capitulo, i_capitulo, titulo):
    tag = {'title':       str(i_capitulo) + ' ' + d_capitulo['song'],
           'artist':      d_capitulo['singer'],
           'album':       d_capitulo['album'],
           'Track':       i_capitulo,
           'Genre':       'Ebook',

           # estos los identifica el picard
           'Date':        '07/07/2021',
           'Subtitle':    'subtitulo',
           'language':    d_capitulo['language'],
           'Comment':     'comm',

           'year':        '2023',
           'Description': 'DESCIPTION',
           'releasetime': '07/07/2021',
           'origyear':    '07/07/2021'
           }

    pa = 'data_out/_images/hi/{}.jpg'.format(titulo)

    return tag, pa

########### OTHERS ####################


def get_largo_capitulos(ll, n_caps=25):
    """
 distribución de cápsulas por capítulo
    :param ll:
    :param n_caps:
    :return:
    """
    nn = ll // n_caps
    kk = ll - (25 * nn)  # estos los distribuimos
    res = sorted([(nn + (1 if x < kk else 0)) for x in range(n_caps)])

    print('Check ok: ', sum(np.array(res)) == ll)
    return res


def get_df_capitulos(caps):
    def repe(x, n):
        return [x for i in range(n)]

    res = get_largo_capitulos(len(caps))
    r2 = [repe(i + 1, n) for i, n in zip(range(len(res)), res)]
    flat_list = [item for sublist in r2 for item in sublist]
    df_b = pd.DataFrame({'i_caps': range(len(caps)), 'capitulo': flat_list, 'txt': caps})

    return df_b


def get_dic_capitulos(df_caps):
    key = 'capsulas'
    dd = {}
    for i, r in df_caps.iterrows():
        cap = r.capitulo
        u = r.txt
        if cap in dd:
            dd[cap][key] = dd[cap][key] + [u]
        else:
            dii = {key: [u]}
            dd[cap] = dii
    return dd


def update_di_capi(di_caps, capitulos_titles, d, titulo):
    """
pone en cada capítulo la información de la "cancion"
    :param di_caps:
    :param capitulos_titles:
    :param d:
    :param titulo:
    """
    for i, capi_title in enumerate(capitulos_titles):
        i_ = i + 1
        uu = di_caps[i_]

        uu['song'] = capi_title
        uu['album'] = d['fakeTitle']
        uu['singer'] = d['fakeAuthor']
        uu['path_cover'] = 'data_out/_images/hi/' + titulo + '.jpg'
        uu['mp3_name'] = str(i_).zfill(2) + ' - ' + capi_title + '.mp3'
        uu['language'] = d['idioma']


def procesa_capitulo(d_capitulos, i_capitulo, titulo, path_book, model, speaker,
                     debug_mode=False, speakers=None, lan='en'):
    import time
    d_capitulo = d_capitulos[i_capitulo]

    path_mp3 = path_book + d_capitulo['album'] + ' - ' + d_capitulo['singer'] + '/'
    make_folder(path_mp3)
    tag, img_path = get_mp3_tag(d_capitulo, i_capitulo, titulo)

    t = inicia('Sintentizando capsula {}'.format(d_capitulo['mp3_name']))
    path_ch = make_folder(path_book + str(i_capitulo).zfill(2))
    au_acc = AudioSegment.silent(100)
    n_caps = len(d_capitulo['capsulas'])
    if debug_mode:
        n_caps = 3

    for k in range(n_caps):
        capsula = d_capitulo['capsulas'][k]
        if debug_mode:
            capsula = capsula[:100]

        # elegimos voz aleatoria en cada tramo
        if speakers is not None:
            import random
            speaker = random.choice(speakers)

        au_capsula = wav_generator(capsula, speaker, k, path_ch, model,
                                   n_caps=str(n_caps), i_capitulo=i_capitulo, lan=lan)
        print(str(k))
        display(au_capsula)
        au_acc = au_acc + au_capsula + AudioSegment.silent(450)

    name_ = path_mp3 + d_capitulo['mp3_name']
    print('** Exportando el final a {}'.format(name_))
    au_acc.export(name_, format="mp3", id3v2_version='3',
                  tags=tag, cover=img_path).close()
    tiempo = tardado(t)

    # actualizamos el json con la info de cuánto tardó en leer
    t2 = time.strftime('%H:%M:%S', time.gmtime(tiempo))
    d_capitulo['elapsed'] = t2
    path_json = 'data_out/{}/{}'.format(titulo, CONTENT_JSON)
    json_update({i_capitulo: d_capitulo}, path_json)


def elige_libros_aleatorios(n,
                            ruta_libros=r"c:\Users\milen\Desktop\del drive\NYT Best Sellers/"):
    import random
    lista_libros = lista_files_recursiva(ruta_libros, 'epub', recursiv=True)
    return random.sample(lista_libros, n)

################## IA ####################


def genera_titulo_openAI(di):
    """
genera un título con la API de OpenAI
    """
    prompt = ''' 
    Quiera que me generaras un título de libro, no más de 8 palabras, combinando con sentido las palabras que te daré, considerando su peso
    ###
    '''
    import openai
    # las keys y contraseñas se
    openai.api_key = OPENAI_API_KEY

    print('** Generando título con OpenAI')
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': json.dumps(di)}]

    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        temperature=0.9,
        messages=messages,
        max_tokens=50,  # maximo de palabras en la respuesta
        # stop=['###'] # para que no siga generando
    )
    res = response.choices[0].message.content
    # quitamos las comillas si es que las puso
    res = res.replace('"', '')
    print('      Título generado: ', res)
    return res


def detecta_ini(texto, nrow=50):

    import re
    from utils_chat import haz
    partes, df = divide_texto(texto, r'\n')

    df_ini = df.head(nrow)

    txt = ''
    for i in range(nrow):
        txt += str(i) + ' | ' + df_ini.loc[i, 'parte'][:50]+'...\n'

    prompt = f"""
    Te daré las primeras palabras de cada párrafo de un libro y su número correlativo. 
    Quiero que me digas en qué número de párrafo empieza la sección principal de libro.
    Muchas veces tiene como título con algo como "Capítulo 1" o "parte primera " o "I".
    No me interesan las secciones de introducción, índices, prefacios, prólogos, etc.
    Dame sólo el número entre triple comillas y que me digas el título de sección que has identificado (como "Parte I")
    Ejemplo: ```12```

    ### Inicio de párrafos

    {txt}
    """
    r = haz(prompt)
    print(f'{r}')
    # Extraemos lo que está entre triple comillas
    ini = int(re.findall(r'```(.*)```', r)[0])
    # mostramos el párrafo previo, el elegido y el siguiente
    for i in range(ini - 1, ini + 2):
        print(str(i) + ' | ' + df.loc[i, 'parte'][:70] + '...')

    return ini


def detecta_fin(texto, i_chunk, nrow=50):

    import re
    from utils_chat import haz
    partes, df = divide_texto(texto, r'\n')
    if i_chunk == -1:
        df_chunk = df[-nrow:]
    else:
        df_chunk = df[i_chunk*nrow:(i_chunk+1)*nrow]  # si i_chunk=-1, df_chunk=df[-50:0],

    print(f' Cogemos los índices {df_chunk.index[0]} a {df_chunk.index[-1]}')
    # print(df_chunk)
    txt = ''
    # recoremos las filas de df_chunk
    for j, row in df_chunk.iterrows():
        # print(f'i: {i}')
        # j = row.index
        # print(f'j: {j}')
        # txt += str(j) + ' | ' + df_end.loc[j, 'parte'][:30]+'...\n'
        txt += str(j) + ' | ' + row.parte[:30] + '...\n'

    prompt = f"""
    Te daré las primeras palabras de cada párrafo del final de un libro y sus número correlativo. 
    Quiero que me digas en qué número de párrafo empiezan secciones que no son parte de la historia.
    Las reconocerás por el título como referencias, agradecimientos, postlogo, notas, etc.
    Quiero que entregues el nombre se la sección encontrada y el número de párrafo. 
    El número debe estar entre comillas triples.
    
    Ejemplo: <titulo>, nº ```12```

    ### Párrafos finales

    {txt}
    """
    print(prompt)
    r = haz(prompt, max_tokens=10, temp=0.1)
    # Extraemos lo que está entre triple comillas
    print(f'Respuesta: [{r}]')
    fin = int(re.findall(r'```(.*)```', r)[0])

    # si el fin es igual al largo del libro, devolvemos -1
    if fin == len(df)-1:
        print('El fin es el último párrafo')
        return -1

    # mostramos el párrafo previo, el elegido y el siguiente
    for i in range(fin - 1, fin + 2):
        print(str(i) + ' | ' + df.loc[i, 'parte'][:70] + '...')

    return fin
