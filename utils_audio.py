from utils_base import win_exe
from utils_io import files_remove
import yaml


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


def tts_google(texto, filename, voice_name, gender='F'):
    # pip install --upgrade google-cloud-texttospeech
    # lista de voces
    # https://cloud.google.com/text-to-speech/docs/voices?hl=es-419
    # https://cloud.google.com/text-to-speech/pricing?hl=es
    from google.cloud import texttospeech
    import os
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "conf/secrets_google.json"

    code = voice_name[:5]
    if gender == 'F':
        gender = texttospeech.SsmlVoiceGender.FEMALE
    else:
        gender = texttospeech.SsmlVoiceGender.MALE

    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=texto)
    # Build the voice request, select the language code ("en-US")
    # ****** the NAME
    # and the ssml voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code=code,
        name=voice_name,
        ssml_gender=gender)

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3)

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    # The response's audio_content is binary.
    with open(filename, 'wb') as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print(f'Audio content written to file "{filename}"')

    registra_tts_uso(filename, texto, 'google')


def registra_tts_uso(filename, texto, motor):
    import datetime
    import os
    n = len(texto)
    hoy = datetime.datetime.now().strftime("%Y-%m-%d")
    yearmonth = datetime.datetime.now().strftime("%Y-%m")
    d = {'fecha': hoy, 'n': n, 'filename': filename, 'texto': texto}

    path = 'data_med/reg_uso.yml'
    # si no existe el archivo lo creamos
    if not os.path.exists(path):
        d_total = {}
    else:
        # leemos el archivo
        with open(path, 'r', encoding='utf8') as infile:
            d_total = yaml.load(infile, Loader=yaml.FullLoader)
    # si no existe el motor lo creamos
    if motor not in d_total:
        d_total[motor] = {}
    # si no existe el año lo creamos
    if yearmonth not in d_total[motor]:
        d_total[motor][yearmonth] = []
    # agregamos 'd'
    d_total[motor][yearmonth].append(d)

    # guardamos
    with open(path, 'w', encoding='utf8') as outfile:
        yaml.dump(d_total, outfile, default_flow_style=False, allow_unicode=True)
