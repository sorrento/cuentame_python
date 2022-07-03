import os
import numpy as np
import pandas as pd
import requests
import json

from IPython.core.display import display
from IPython.lib.display import Audio
from pydub import AudioSegment

from secret_keys import *
from u_base import get_now_format, inicia, tardado, json_read, make_folder
from u_io import lista_files_recursiva, fecha_mod, get_filename, txt_read, txt_write
from u_plots import plot_hist
from u_text import numero_a_letras
from u_textmining import get_candidatos_nombres_all, pick

SUMMARIES_JSON = 'data/summaries.json'


def seleccion_txt(path):
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

    # quitamos los numeros:
    digits = [x for x in ejj.index if x.isdigit()]
    ejj = ejj[~ejj.index.isin(digits)]

    # quitamos las palabras que se usaron como fake author
    if l_authors is None:
        ejj2 = ejj
    else:
        ejj2 = ejj[~ejj.index.isin([x.lower() for x in l_authors])]

    return pick(ejj2, 15, 3, 'value')


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
    nombres = list(df_names.index)

    l_title = get_fake_title(vector_matrix, vocab, i, nombres)
    fake_title = ' '.join(l_title)

    di["fakeAuthor"] = fake_authors
    di["fakeTitle"] = fake_title
    di["path"] = file
    di["listo"] = False
    di["i"] = i
    di['names'] = nombres

    return di, d_count


def get_fakes(doc_list, files, vector_matrix, vocab, lang):
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
        di_fake, di_count = get_book_summary(i, files, doc_list, vector_matrix, vocab)
        # print(di_fake)
        di_fake['idioma'] = lang
        di_fakes[i] = di_fake
        di_counts[i] = di_count

    tardado(t)

    return di_fakes, di_counts


def get_books(path):
    files = seleccion_txt(path)
    doc_list = [txt_read(x) for x in files]
    return doc_list, files


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
            if verbose: print('***caemos dentro:', n_fut)
            g, n_acc = agrega(grupos, largos, g, n_fut, i, d, partes)

    df['capsula'] = grupos
    if verbose: plot_hist(largos, 45)

    return d


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
            if verbose: print('  ** >> pasamos', n_fut)
            largos.append(n_new)  # cerramos el anterior
            g = g + 1
            agg(grupos, g, d, i, partes)
            n_acc = n_new
        else:
            agg(grupos, g, d, i, partes)
            n_acc = n_fut

    df['capsula'] = grupos
    if verbose: plot_hist(largos, 45)

    if verbose:
        print('** Máximo largo: {}, mínimo: {}'.format(str(max(largos)), str(min(largos))))
    return d


def get_image_path(file):
    oo = file.split('\\')[:-1]
    oo.append('cover.jpg')
    return '/'.join(oo)


def get_headers():
    return {
        'X-Parse-Application-Id': X_PARSE_APPLICATION_ID,
        'X-Parse-REST-API-Key':   X_PARSE_REST_API_KEY,
        'Content-Type':           'application/json'
    }


def update_status(objectId):
    url = "https://parseapi.back4app.com/classes/WordCorpus/" + objectId
    payload = {'status': True}
    header = get_headers()
    response = requests.put(url, data=json.dumps(payload), headers=header)
    print(response.text)
    return response.status_code


class Back4App:
    def get_sentance(self):
        header = get_headers()
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


def upload_lib_summary(j):
    url = "https://parseapi.back4app.com/classes/librosSum/"

    j.pop('path')
    j.pop('listo')
    j.pop('i')

    v = str(j).replace('\'', '\"').encode('utf-8')
    header = get_headers()
    data = requests.post(url, data=v, headers=header)

    print(data.json())


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


def audio_save(au, name, path, mp3=True, show=False, tag=None):
    tem = 'temp.wav'
    wa = None
    if mp3:
        ext = '.mp3'
    else:
        ext = '.wav'

    no = path + '/' + name + ext
    print('Guardando ', no)

    if mp3:
        with open(tem, 'wb') as f:
            f.write(au.data)

        wa = AudioSegment.from_wav(tem)
        # wa.export(no, format="mp3", tags=tag).close()
        wa.export(no, format="mp3").close()
        os.remove(tem)
    else:
        with open(no, 'wb') as f:
            f.write(au.data)

    if show:
        display(au)

    return wa


def speakers_test(model, put_accent=True, sample_rate=48000, put_yo=True,
                  txt='Formalmente, desde el Acuerdo Marco "Aurora" de 1953, los centros pertenecientes a la red '
                      'mundial debían "trabajar en plena colaboración académica y humana, compartiendo los avances '
                      'tanto en conocimientos fundamentales como en técnicas.'):
    print(txt)
    sps = [x for x in model.speakers if x != 'random']
    for sp in sps:
        # print(sp)
        audio = model.apply_tts(text=reemplaza_nums(txt),
                                speaker=sp,
                                sample_rate=sample_rate,
                                put_accent=put_accent,
                                put_yo=put_yo
                                )
        au = Audio(audio, rate=sample_rate)

        tag = {'title': 'Voice Test: ' + sp, 'artist': sp}
        audio_save(au, 'data_out/wav/test_' + sp, show=True, tag=tag)


def lee(model,
        txt='Formalmente, desde el Acuerdo Marco "Aurora" de 1953, los centros pertenecientes a la red mundial debían '
            '"trabajar en plena colaboración académica y humana, compartiendo los avances tanto en conocimientos '
            'fundamentales como en técnicas.', sample_rate=48000, put_accent=True, put_yo=True):
    audio = model.apply_tts(text=txt,
                            speaker='es_1',
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo
                            )
    au = Audio(audio, rate=sample_rate)
    # guarda_wav(au, 'te_' + sp)
    return au


def reemplaza_nums(new_string):
    import re
    # new_string = 'Rose67lilly78Jasmine228Tulip'
    new_result = re.findall('[0-9]+', new_string)

    if len(new_result) == 0:
        return new_string

    dic = {x: numero_a_letras(int(x)) for x in new_result}
    for key, value in dic.items():
        # Replace key character with value character in string
        new_string = new_string.replace(key, value)
    return new_string


def wav_generator(txt, voz, i_cap, path, model, write_txt=True,
                  sample_rate=48000, put_accent=True, put_yo=True):
    t = inicia(' i = {}'.format(i_cap))
    print(txt[0:30])

    name = str(i_cap).zfill(4) + '_' + voz
    audio = model.apply_tts(text=reemplaza_nums(txt),
                            speaker=voz,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo
                            )
    au = Audio(audio, rate=sample_rate)
    au_seg = audio_save(au, name, path)

    if write_txt:
        txt_write(path + '/' + name, txt)

    tardado(t)

    return au_seg


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
    for i, e in enumerate(capitulos_titles):
        i_ = i + 1
        uu = di_caps[i_]

        uu['song'] = str(i_) + '. ' + e
        uu['album'] = d['fakeTitle']
        uu['singer'] = d['fakeAuthor']
        uu['path_cover'] = 'data_out/images/hi/' + titulo + '.jpg'
        uu['mp3_name'] = str(i_).zfill(2) + ' - ' + e + '.mp3'


def get_mp3_tag(dd, i_cap, titulo):
    tag = {'title':       dd['song'], 'artist': dd['singer'], 'album': dd['album'],
           'Track':       i_cap,
           'Genre':       'Ebook',

           # estos los identifica el picard
           'Date':        '07/07/2021',
           'Subtitle':    'subtitulo',
           'language':    dd['lan'],
           'Comment':     'comm',

           'year':        '2023',
           'Description': 'DESCIPTION',
           'releasetime': '07/07/2021',
           'origyear':    '07/07/2021'
           }

    pa = 'data_out/images/hi/{}.jpg'.format(titulo)

    return tag, pa


def procesa_capitulo(j, i_cap, titulo, path_book, model):
    dd = j[i_cap]

    dd['lan'] = 'ES'  # todo tiene que venir
    tag, img_path = get_mp3_tag(dd, i_cap, titulo)

    t = inicia('Sintentizando cap {}'.format(dd['mp3_name']))
    path_ch = make_folder(path_book + str(i_cap).zfill(2))
    au_acc = AudioSegment.silent(100)
    for k in range(len(dd['capsulas'][:3])):
        cap = dd['capsulas'][k]
        au_capsula = wav_generator(cap[:100], 'es_1', k, path_ch, model)
        au_acc = au_acc + au_capsula + AudioSegment.silent(300)
    au_acc.export(path_book + dd['mp3_name'], format="mp3", id3v2_version='3',
                  tags=tag, cover=img_path).close()
    tardado(t)


def get_book_datas(pat):
    from PIL import Image
    j = json_read(SUMMARIES_JSON)
    titles = sorted(list(j.keys()))
    titulo = [x for x in titles if pat in x][0]
    print(titulo)
    d = j[titulo]
    texto = txt_read(d['path'])
    im = Image.open(get_image_path(d['path']))

    return texto, im, titulo, d
