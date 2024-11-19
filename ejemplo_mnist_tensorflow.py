import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Partir de los datos de MNIST, 60.000 imágenes manuscritas de números entre 0-9 
# en formato 28x28 en B/N para entrenar una red neuronal y ver si es capaz de 
# reconocer los números en los datos de pruebas  

# Cargar los datos de MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocesar los datos, normalizamos los datos entre 0-1
x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0

# Definir, compilar y entrenar el modelo
# Definimos el modelo secuencial
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilamos el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamos el modelo
history = model.fit(x_train, y_train, epochs=5)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc) # Test accuracy: 0.932200014591217

# Graficar la pérdida
plt.plot(history.history['loss'])
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend(['Entrenamiento'], loc='upper right')
plt.show()

# Graficar la precisión
plt.plot(history.history['accuracy'])
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend(['Entrenamiento'], loc='upper left')
plt.show()

# Ejemplo de imagen 
#print(x_train[0]) 

# Obtener una imagen. Vector de 784 valores pasarlo a 28x28
image = x_train[0]
image = np.reshape(image, (28, 28))  # Reshape to 28x28

# Mostrar la imagen
plt.imshow(image, cmap='gray')
plt.title(f'Imagen de MNIST: {y_train[0]}')
plt.show()
