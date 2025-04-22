# Bibliotecas
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from PIL import Image

# Configuración
data_dir = "/home/camc/MisProyectos/espectros_aves"
categories = ["ave1", "ave2"]
img_size = (224, 224)

# Función para cargar los datos
def load_data(data_dir, categories, img_size):
    data = []
    labels = [] 
    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            print(f"Advertencia: La carpeta {category} no existe.")
            continue
        for file in os.listdir(category_path):
            try:
                img_path = os.path.join(category_path, file)
                with Image.open(img_path).resize(img_size) as img:
                    img = np.array(img)
                    data.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Error cargando {file}: {e}")
    return np.array(data), np.array(labels)

# Cargar datos
data, labels = load_data(data_dir, categories, img_size)

# Verificar que data y labels tengan el mismo tamaño
if len(data) != len(labels):
    raise ValueError(f"Número de imágenes ({len(data)}) y etiquetas ({len(labels)}) no coinciden.")

# Normalizar imágenes con preprocess_input de MobileNetV2
data = preprocess_input(data.astype("float32"))

# Convertir etiquetas a one-hot encoding
labels = to_categorical(labels, num_classes=len(categories))

# División en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# Modelo MobileNetV2 (modelo pre-entrenado)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Congelar capas preentrenadas
base_model.trainable = False

# Descongelas lasultimas 10 capas
for layer in base_model.layers[-5:]:  
    layer.trainable = True  

# Construccion del modelo
model = Sequential([
    base_model,
    GlobalAveragePooling2D(), 
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(len(categories), activation="softmax") 
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001),  
              loss="categorical_crossentropy",  
              metrics=["accuracy"])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss',  
                               patience=5,
                               restore_best_weights=True,
                               verbose=1)        

# Entrenamiento del modelo
history = model.fit(x_train, y_train,  
                    validation_data=(x_test, y_test), 
                    epochs=50,
                    batch_size=32, 
                    callbacks=[early_stopping])

# Evaluar modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Precisión en el conjunto de prueba: {test_acc:.2f}")

# Guardar el modelo entrenado
model.save("modelo_CNNaves_MobileNetV2.keras")

# Visualizar el resumen del modelo en consola
model.summary()

# Visualización gráfica del modelo
plot_model(model, to_file='modelo_architecture_MovileNetV2.png', show_shapes=True, show_layer_names=True)

# Función para graficar la precisión y pérdida
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Entrenamiento")
    plt.plot(history.history["val_accuracy"], label="Validación")
    plt.xlabel("Épocas")
    plt.ylabel("Precisión")
    plt.title("Evolución de Precisión")
    plt.legend()
    plt.grid()

    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validación")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title("Evolución de Pérdida")
    plt.legend()
    plt.grid()
    
    # Guardar la imagen antes de mostrarla
    plt.savefig("grafica_entrenamiento_MovileNetV2.png", dpi=300, bbox_inches='tight')

# Mostrar gráficos de entrenamiento
plot_training_history(history)
