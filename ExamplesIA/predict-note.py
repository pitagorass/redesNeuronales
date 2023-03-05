import tensorflow as tf
import numpy as np

# Construir el vocabulario de notas de piano
notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
note_to_idx = {note: idx for idx, note in enumerate(notes)}
idx_to_note = {idx: note for idx, note in enumerate(notes)}
num_classes = len(notes)

# Convertir las secuencias de notas en secuencias de índices
sequences = ['C C# D D# E F F# G G# A A# B', 'C C# D D# E F F# G G# A A#']
input_data = []
for seq in sequences:
    input_seq = [note_to_idx[note] for note in seq.split()]
    input_data.append(np.array(input_seq))
input_data = np.array(input_data)

# Construir el modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, input_shape=(None, 1)))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Preparar los datos de entrada para el modelo
input_data = np.expand_dims(input_data, axis=1)
output_data = np.roll(input_data, -1, axis=1)
output_data[:, -1] = 0



# Entrenar el modelo
model.fit(input_data, output_data, epochs=100)

# Hacer una predicción para la siguiente nota en la primera secuencia
prediction = model.predict(input_data[0][None, :, :])
predicted_note_idx = np.argmax(prediction[0, -1, :])
print('Input sequence:', sequences[0])
print('Predicted next note:', idx_to_note[predicted_note_idx])
