import os


def getmtime(filename):
    """Return the last modification time of a file, reported by os.stat()."""
    return os.stat(filename).st_mtime


def getatime(filename):
    """Return the last access time of a file, reported by os.stat()."""
    return os.stat(filename).st_atime


def getctime(filename):
    """Return the metadata change time of a file, reported by os.stat()."""
    return os.stat(filename).st_ctime


def lista_files_recursiva(path, ext, with_path=True, drop_extension=False, recursiv=True, filter=None):
    """
devuelve la lista de archivos en la ruta, recursivamente, de la extensión especificada. la lista está ordenada por fecha
de modificación
    :param filter: deja sólo los resultados que contengan este string
    :param drop_extension:
    :param with_path:
    :param path:
    :param ext:
    :return:
    """
    import glob

    if recursiv:
        lista = glob.glob(path + '**/*.' + ext, recursive=recursiv)
    else:
        lista = glob.glob(path + '/*.' + ext, recursive=recursiv)

    lista = sorted(lista, key=getmtime, reverse=True)

    if not with_path:
        lista = [get_filename(x) for x in lista]

    if drop_extension:
        lista = [remove_extension(x) for x in lista]

    if filter is not None:
        lista = [x for x in lista if filter in x]

    return sorted(lista)


def remove_extension(filename):
    return '.'.join(filename.split('.')[:-1])


def fecha_mod(file):
    """
entrega la fecha de modificación de un archivo como un número yyyymmdd
    :param file:
    :return:
    """
    import datetime
    dt = datetime.datetime.fromtimestamp(getmtime(file))
    return int(dt.strftime('%Y%m%d'))


def get_filename(path, remove_ext=False):
    """
obtiene el nombre del fichero desde el path completo
    :param remove_ext: quitar la extension
    :param path:
    :return:
    """
    file = os.path.basename(path)
    if remove_ext:
        file = remove_extension(file)
    return file


def txt_read(file_path):
    """
lee fichero de texto
    :param file_path:
    :return:
    """
    import os
    try:
        isfile = os.path.isfile(file_path)
    except Exception as e:
        print('No es file:{}\n{}'.format(file_path, str(e)))
        isfile = False

    if isfile:
        # open text file in read mode
        text_file = open(file_path, "r", encoding='utf-8')

        # read whole file to a string
        data = text_file.read()

        # close file
        text_file.close()
        return data
    else:
        return ''


def txt_write(file_path, txt):
    txt_ = file_path + '.txt'
    print('  ** Guardando ', txt_)
    text_file = open(txt_, "w", encoding='utf-8')
    text_file.write(txt)
    text_file.close()


def files_remove(path, ext, recur=False):
    import os
    import glob
    # Get a list of all the file paths that ends with .txt from in specified directory
    # fileList = glob.glob('C://Users/HP/Desktop/A plus topper/*.txt')
    b = ''
    if recur:
        b = '/**'

    fileList = glob.glob(path + b + '/*.' + ext)
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except Exception as e:
            print("Error while deleting file : ", filePath)


def has_extension(path):
    res = '.' in path[-5:-2]
    # print('{} has extensio: {}'.format(path, res))
    return res
