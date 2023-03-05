""" import IPython.display as ipd
import numpy

sr = 22050 # sample rate
T = 0.5    # seconds
t = numpy.linspace(0, T, int(T*sr), endpoint=False) # time variable
x = 0.5*numpy.sin(2*numpy.pi*440*t)              # pure sine wave at 440 Hz
ipd.Audio(x, rate=sr, autoplay=True) # load a NumPy array """
import numpy as np
import IPython.display as ipd
from scipy.io import wavfile

seconds = 1
sample_rate = 4000
t = np.arange(int(seconds * sample_rate)) / sample_rate
x = np.sin(2 * np.pi * 440 * t)

# Guardar el archivo de audio
wavfile.write('sound.wav', sample_rate, x)

# Reproducir el archivo de audio con el reproductor predeterminado
ipd.Audio('sound.wav')