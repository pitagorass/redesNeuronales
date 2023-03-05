import IPython.display as ipd

# Carga un archivo de audio
audio_file = 'C:/Users/santiago.vargas/OneDrive/Documentos/Pitagoras/Programacion/Repositorios/IA/musicExternal/MusicMidi/maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.wav'

# Reproduce el archivo de audio
ipd.Audio(audio_file, autoplay=True)
