""" This tutorial shows you how to generate musical notes using a simple recurrent 
neural network (RNN). You will train a model using a collection of piano MIDI files 
from the MAESTRO dataset. Given a sequence of notes, your model will learn to predict 
the next note in the sequence. You can generate longer sequences of notes by calling 
the model repeatedly. """
import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000.0
# we get our midi files from our folder
data_dir = pathlib.Path(
    'C:/Users/santiago.vargas/OneDrive/Documentos/Pitagoras/Programacion/Repositorios/IA/musicExternal/MusicMidi/maestro-v3.0.0')
filenames = glob.glob(str(data_dir/'**/*.mid*'))
# The dataset contains about 1,200 MIDI files.
print('Number of files:', len(filenames))

""" Process a MIDI file
First, use pretty_midi to parse a single MIDI file and inspect the format of the notes. 
If you would like to download the MIDI file below to play on your computer, you can do 
so in colab by writing files.download(sample_file). """
sample_file = filenames[1]
print(sample_file)

# attempt to generate a PrettyMIDI object for the sample MIDI file
pm = pretty_midi.PrettyMIDI(sample_file)
print(pm)
# Play the sample file. The playback widget may take several seconds to load.
def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
    waveform = pm.fluidsynth(
        fs=_SAMPLING_RATE, sf2_path=r'C:/ProgramData/soundfonts/default.sf2')
    print('Wavefrom', waveform)
    # Take a sample of the generated waveform to mitigate kernel resets
    waveform_short = waveform[:int(seconds*_SAMPLING_RATE)]
    # waveform_short = waveform[:int(100)]
    print('waveform_short', waveform_short)
    return display.Audio(data=waveform_short, rate=int(_SAMPLING_RATE))

display_audio(pm)


#Do some inspection on the MIDI file. What kinds of instruments are used?
print('Number of instruments:', len(pm.instruments))
instrument = pm.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
print('Instrument name:', instrument_name)