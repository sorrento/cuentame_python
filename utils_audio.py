from utils_base import win_exe
from utils_io import files_remove
import yaml

voices_azure = ['es-CL-CatalinaNeural',
                'es-CL-LorenzoNeural',
                'es-ES-AlvaroNeural',
                'es-ES-ElviraNeural',
                'es-ES-HelenaNeural',
                'es-ES-AbrilNeural',
                'es-ES-ArnauNeural',
                'es-ES-DarioNeural',
                'es-ES-EliasNeural',
                'es-ES-EstrellaNeural',
                'es-ES-IreneNeural',
                'es-ES-LaiaNeural',
                'es-ES-LiaNeural',
                'es-ES-NilNeural',
                'es-ES-SaulNeural',
                'es-ES-TeoNeural',
                'es-ES-TrianaNeural',
                'es-ES-VeraNeural'
                ]


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


def get_random_voice_azure():
    import random
    voice_name = random.choice(voices_azure)
    return voice_name


def tts_azure(texto, filename, voice_name=None):

    # pip install azure-cognitiveservices-speech

    from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, ResultReason, CancellationReason
    from conf.secrets_key import TOKEN_AZURE_SPEECH
    # si la voz no está definida, elegimos una al azar de la lista
    if voice_name is None:
        voice_name = get_random_voice_azure()
        print(f'No se ha definido la voz, se elige una al azar: {voice_name}')

    region = 'westeurope'
    speech_config = SpeechConfig(subscription=TOKEN_AZURE_SPEECH, region=region)
    speech_config.speech_synthesis_voice_name = voice_name
    # audio_config = AudioOutputConfig(use_default_speaker=True)

    audio_output = AudioConfig(filename=filename)

    # Crea un sintetizador de voz con la configuración de voz y audio.
    speech_synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)

    # Sintetiza el texto en un archivo de audio sin reproducirlo.
    speech_synthesis_result = speech_synthesizer.speak_text_async(texto).get()

    if speech_synthesis_result.reason == ResultReason.SynthesizingAudioCompleted:
        print(f"Se ha sintetizado el habla para el texto [{texto[:30]}] y se ha guardado en {filename}")
        registra_tts_uso(filename, texto, 'azure')
    elif speech_synthesis_result.reason == ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Se canceló la síntesis de voz: {}".format(cancellation_details.reason))
        if cancellation_details.reason == CancellationReason.Error:
            if cancellation_details.error_details:
                print("Detalles del error: {}".format(cancellation_details.error_details))
                print("¿Configuraste correctamente la clave de recurso de voz y los valores de la región?")


def genera_presentacion(voice):
    # genera un texto, por ejemplo:
    #  voice='es-ES-AlvaroNeural' -> hola, soy alvaro, de España
    txt = f'Hola, soy {voice.split("-")[2]}, de '
    pais = voice.split("-")[1]
    if pais == 'ES':
        pais = 'España'
    elif pais == 'CL':
        pais = 'Chile'
    txt += pais
    return txt


def azure_samples():
    for voice in voices_azure:
        filename = f'data_out/azure/{voice}.mp3'
        texto = genera_presentacion(voice)
        tts_azure(texto, filename, voice)


def registra_tts_uso(filename, texto, motor):
    """
    Registra el uso de los tts en un archivo yaml, para sabber cuántos caracteres hemos usado en el mes
    Google tiene un límite gratis de 1M
    Azure tiene un límite gratis de 0.5M
    """
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


def get_audiosegment(mp3_file):
    """
    Devuelve un objeto AudioSegment, lo hacemos robusto de manera que si el mp3 tiene problemas con las 
    cabeceras, lo transforma a wav y lo leemos como segmento
    """
    from pydub import AudioSegment
    import os
    wav_file = 'temp.wav'
    if os.path.exists(wav_file):
        os.remove(wav_file)
    # abs path
    wav_file = os.path.join(os.getcwd(), wav_file)
    mp3_file = os.path.join(os.getcwd(), mp3_file)
    
    print(f'*** Segmento de audio: {mp3_file}   ***')
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        print('   OK')        
    except Exception as e:

        print('   Como WAw')
        cmd = f'ffmpeg -i "{mp3_file}" "{wav_file}"'
        win_exe(cmd)
        audio = AudioSegment.from_wav(wav_file)

    return audio