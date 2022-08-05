from base import win_exe
from ut.io import files_remove


def wav2mp3(titulo):
    path = 'data_out/wav/%s' % titulo
    cmd = 'wav2mp3.exe ' + '"' + path + '"'
    res = win_exe(cmd)
    all_converted_ok = False  # todo implementar check por ejemplo contantdo los "converted"
    if all_converted_ok:
        files_remove(path, 'wav')
    else:
        print('** NO se han borrado los wav porque parece que no se ha convertido ok')
    return res
