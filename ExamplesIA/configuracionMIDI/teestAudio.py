import matplotlib.pyplot as plt
import librosa.display


x, sr = librosa.load('C:/Users/santiago.vargas/OneDrive/Documentos/Pitagoras/Programacion/Repositorios/IA/musicExternal/MusicMidi/maestro-v3.0.0/2018/MIDI-Unprocessed_Schubert7-9_MID--AUDIO_15_R2_2018_wav.wav')

print(x.shape)
print(sr)
librosa.display.waveshow(sr)

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
