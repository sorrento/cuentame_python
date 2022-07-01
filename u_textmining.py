import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf_preprocessing(doc_list, kwargs):
    """Tf-IDF vectorization of corpus

    Parameters
    ----------
    doc_list : iterable
        Iterable list of texts
    kwargs : dict
        Dictionary with hyperparameters

    Returns
    -------
    vector_matrix, vocab, doc_freq:
        vector_matrix : NumPy array
            TF-IDF value matrix
        vocab : dict
            Token vocabulary dictionary
        doc_freq : NumPy array
            Document frequency for each token in vocab
    """

    if kwargs.get("tfidf_stop_words"):
        if kwargs.get("tfidf_strip_accents"):
            accents = 'ascii'
            with open('./data/stopwords_ascii.txt', 'r', encoding='utf-8') as fs:
                stopwords = fs.read().splitlines()
        else:
            accents = None
            with open('./data/stopwords.txt', 'r', encoding='utf-8') as fs:
                stopwords = fs.read().splitlines()
    else:
        stopwords = None
        accents = 'ascii' if kwargs.get("tfidf_strip_accents") else None

    text_vectorizer = TfidfVectorizer(
        strip_accents=accents,
        analyzer=kwargs.get("tfidf_analyzer"),
        stop_words=stopwords,
        ngram_range=(kwargs.get("tfidf_ngram_range_min"),
                     kwargs.get("tfidf_ngram_range_max")),
        max_df=kwargs.get("tfidf_max_df"),
        min_df=kwargs.get("tfidf_min_df"),
        norm=None,
    )

    vector_matrix = text_vectorizer.fit_transform(doc_list)
    # Obtain vocabulary and document frequency
    vocab = np.array(text_vectorizer.get_feature_names())
    doc_freq = 1.0 / text_vectorizer.idf_

    return vector_matrix, vocab, doc_freq


def porcentaje_en_mayuscula(texto, word):
    """
cuenta el porcentaje de veces que una palabra aparece con primera letra en mayúscula en el texto
    :rtype: object
    """
    div = texto.split()  # es necesario, porque nombres pueden estar dentro otra palabra, como eva y evaluación
    a = div.count(word)
    b = div.count(word.capitalize())

    n = a + b
    if n != 0:
        res = b / n
    else:
        print(word)
        res = 0
        n = 0

    return res, n


def get_candidatos_a_nombre(doc_list, vector_matrix, vocab, i):
    """
a partir de los resultados del tf idf  devuelve una lista de los que parecen ser nombres comparando las veces que aparecen
en mayúsculas. Tiene el problema que los nombres que aparecen poco en un libro no estarán a menos que se haga el tfidf
con parámetros extremos y salga una matriz muy grande
    :param doc_list:
    :param vector_matrix:
    :param i:
    :return:
    """
    ej = pd.melt(pd.DataFrame(vector_matrix[i, :].todense(), columns=vocab))
    best = ej.sort_values('value', ascending=False).head(70)
    print(sorted(best.variable.to_list()))
    texto = doc_list[i]

    best['n'] = best['variable'].map(lambda x: porcentaje_en_mayuscula(texto, x))
    best['perc_may'] = best.n.map(lambda x: x[0])
    best['n'] = best.n.map(lambda x: x[1])

    best = best.sort_values('perc_may', ascending=False)

    # caso habitual
    seleccion = best[(best.perc_may > .8) & (best.n > 10)]
    if len(seleccion) > 3:
        res = seleccion
    else:
        res = best.head(3)
    return res


def get_candidatos_nombres_all(texto):
    """
la lista de posibles nombres propios, en un df con las veces que aparece en mayúscula y en total
    :param texto:
    :return: también devuelve un diccionario con el conteo de palabras
    """
    import re
    import collections

    ss = split(texto)
    ws = [x for x in ss if x != '']

    # Contamos las que aparecen con mayúscula
    mays = [x for x in ws if re.search('[A-Z]\\w+', x)]
    d_mays = dict(collections.Counter(mays))

    # Contamos las veces que aparecen éstas en minúscula
    mins = [x.lower() for x in d_mays]
    d_all = dict(collections.Counter(ws))
    d_count_total = {x: d_all[x] for x in mins if x in d_all}

    # Unimos en un dataframe
    f1 = pd.DataFrame.from_dict({x.lower(): d_mays[x] for x in d_mays}, orient='index').rename(columns={0: 'n_may'})
    f2 = pd.DataFrame.from_dict(d_count_total, orient='index').rename(columns={0: 'n'})
    r = f1.join(f2)

    r = r.replace(np.nan, 0)
    r['N'] = r['n_may'] + r['n']
    r['ratio'] = r['n_may'] / r['N']
    r = r.sort_values('ratio', ascending=False)

    return r[r.ratio > 0.8].sort_values('N', ascending=False), d_all


def split(texto):
    """
separa un string un palabras de acuerdo a los separadores que defino. No lo hago con tokenizador porque quiero mantener
las que son mayúsculas (quizás sí se puede, no lo sé)
    :param texto:
    :return:
    """
    import re
    # s = texto.split()  # no las corta bien. deja guión inicial, y al final, puntos o comas
    ss = re.split(r'[-,\.\s—¿\?!¡;:…»\(\)”]\s*', texto)
    return ss


def tf_idf_keywords(docs, vector_matrix, vocab, doc_freq, num_keywords):
    """Obtain a list of keywords for each document based on the TF-IDF score.
    For each document it return a list of tuples (word, freq), where word is
    the keyword and freq is the number of occurrences in the document.

    Parameters
    ----------
    docs : iterable
        Iterable object that yields Document objects
    vector_matrix : NumPy array
        NumPy matrix with Tf-Idf values
    vocab : dict
        Dictionary [num. key] -> [keyword]
    doc_freq : NumPy array
        Document frequencies for each term
    num_keywords : int
        Max. number of keywords to be kept for each document

    Returns
    -------
    List of Document objects
    """
    docs_to_update = []
    # Obtain keywords from TfidfVectorizer
    for doc, vect_text in zip(docs, vector_matrix):
        vect_text = vect_text.todense().reshape((-1))
        # Get terms with highest tf-idf score
        pos = vect_text.argsort().tolist()[0][-num_keywords:][::-1]
        # Covert to document-wise term frequency
        vect_text = np.multiply(vect_text, doc_freq)
        vect_text = vect_text[0, pos].tolist()[0]
        # Convert to integer
        vect_text = [int(np.round(score, 0)) for score in vect_text]
        kw = list(zip(vocab[pos], vect_text))
        # Filter numbers and empty occurrences
        kw = [kw_pair for kw_pair in kw if kw_pair[1] > 0
              and not kw_pair[0].translate(str.maketrans('', '', string.punctuation + ' ')) \
            .isnumeric()]
        doc.keywords = kw
        docs_to_update.append(doc)

    return docs_to_update


def pick(df, top, n, var_peso='N'):
    """
de un df, coge las top rows y selecciona n índices de acuerdo al peso dado por la columna var_peso
    :param df:
    :param top:
    :param n:
    :param var_peso:
    :return:
    """

    df = df.head(top)
    noms = [x.capitalize() for x in df.index]
    #     random.choices(noms, weights=nombres.N,k=3) no puedo hacer sin reemplazo
    pesos = df[var_peso] / sum(df[var_peso])
    l = list(np.random.choice(noms, n, False, pesos))

    return l


def palabras_representativas(lista, n_best=3, n_pick=3):
    """
Forma lista de strings, con las n_pick palabras más representativas de cada texto. Escoge n_pick aleatoriamente
de las n_best
    :param lista: lista con textos
    :param n_best:
    :param n_pick:
    :return:
    """
    import pandas as pd
    vector_matrix, vocab = get_word_matrix(lista)

    def oo(i, n_best, n_pick):
        ej = pd.melt(pd.DataFrame(vector_matrix[i, :].todense(), columns=vocab))
        ejj = ej.sort_values('value', ascending=False).set_index('variable')  # preparamos para la funcion pick

        return ' '.join(pick(ejj, n_best, n_pick, 'value'))

    return [oo(i, n_best, n_pick) for i in range(len(lista))]


def get_word_matrix(doc_list):
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
    return vector_matrix, vocab