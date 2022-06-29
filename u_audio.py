from u_base import win_exe
from u_io import files_remove


def wav2mp3(titulo):
    # Es posible hacerlo desde python, pero hay que instalar el ffmpeg y es un poco webiao
    # from pydub import AudioSegment
    # AudioSegment.from_wav("data_out/wav/test_es_0.wav").export("data_out/wav/test_es_0.mp3", format="mp3")
    path = 'data_out/wav/%s' % titulo
    cmd = 'wav2mp3.exe ' + '"' + path + '"'
    res = win_exe(cmd)
    all_converted_ok = False  # todo implementar check por ejemplo contantdo los "converted"
    if all_converted_ok:
        files_remove(path, 'wav')
    else:
        print('** NO se han borrado los wav porque parece que no se ha convertido ok')
    return res