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


def lista_files_recursiva(path, ext):
    """
devuelve la lista de archivos en la ruta, recursivamente, de la extensión especificada. la lista está ordenada por fecha
de modificación
    :param path:
    :param ext:
    :return:
    """
    import glob
    lista = glob.glob(path + '**/*.' + ext, recursive=True)
    lista = sorted(lista, key=getmtime, reverse=True)

    return lista


def fecha_mod(file):
    """
entrega la fecha de modificación de un archivo como un número yyyymmdd
    :param file:
    :return:
    """
    import datetime
    dt = datetime.datetime.fromtimestamp(getmtime(file))
    return int(dt.strftime('%Y%m%d'))


def get_filename(path):
    """
obtiene el nombre del fichero desde el path completo
    :param path:
    :return:
    """
    return os.path.basename(path)


def txt_read(file_path):
    """
lee fichero de texto
    :param file_path:
    :return:
    """
    import os
    if os.path.isfile(file_path):
        # open text file in read mode
        text_file = open(file_path, "r", encoding='utf-8')

        # read whole file to a string
        data = text_file.read()

        # close file
        text_file.close()
        return data


def txt_write(file_path, txt):
    text_file = open(file_path + '.txt', "w", encoding='utf-8')
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
