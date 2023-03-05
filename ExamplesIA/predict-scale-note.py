import numpy as np
import keras
from keras.layers import SimpleRNN, Dense
from keras.models import Sequential

# Crear una secuencia de datos de entrenamiento de notas de la escala de La mayor
notes = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23, 24]
timesteps = 5

# Crear un conjunto de datos de entrenamiento que consiste en secuencias de 5 notas consecutivas
X_train = []
y_train = []
for i in range(len(notes) - timesteps - 1):
    X_train.append(notes[i:i+timesteps])
    y_train.append(notes[i+timesteps])

# Convertir los datos de entrenamiento a arrays de numpy
X_train = np.array(X_train)
y_train = np.array(y_train)

# Crear un modelo secuencial de Keras
model = Sequential()

# Añadir una capa SimpleRNN a la red neuronal
model.add(SimpleRNN(64, input_shape=(timesteps, 1)))

# Añadir una capa densa a la red neuronal
model.add(Dense(1, activation='linear'))

# Compilar el modelo
model.compile(loss='mse', optimizer='adam')

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train.reshape(-1, timesteps, 1), y_train, epochs=100, batch_size=1, verbose=1)

# Crear una secuencia de prueba de notas de la escala de La mayor
X_test = [16, 17, 19, 21, 23]

# Hacer una predicción para la nota siguiente en la secuencia de prueba
prediction = model.predict(np.array(X_test).reshape(1, timesteps, 1))[0][0]

# Redondear la predicción a la nota más cercana en la escala de La mayor
note = int(round(prediction / 2.0) * 2)

# Imprimir la nota predecida
print(f'Nota predecida: {note}')