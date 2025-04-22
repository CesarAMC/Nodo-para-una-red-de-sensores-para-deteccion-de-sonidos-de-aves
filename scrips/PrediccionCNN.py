# Bibliotecas
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # mobilenet_v2 - resnet50 - efficientnet
import csv

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("/home/camc/MisProyectos/modelo_CNNaves_MobileNetV2.keras") # MobileNetV2 ResNet50 EfficientNetB0

# Clases del modelo
clases = ["Ave1_Tragon", "Ave2_Cerquero"]

# Umbral de confianza (% de acierto mínimo para considerar la predicción válida)
UMBRAL_CONF = 0.9

# Función para preprocesar imágenes
def procesar_imagen(ruta_imagen):
    img = image.load_img(ruta_imagen, target_size=(224, 224))  # Cargar imagen con tamaño adecuado
    img_array = image.img_to_array(img)  # Convertir a array de NumPy
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión batch
    img_array = preprocess_input(img_array) 
    return img_array

# Función para hacer la predicción, mostrarla en la terminal y guardarla en CSV
def predecir(ruta_imagen, archivo_csv):
    img_procesada = procesar_imagen(ruta_imagen)
    prediccion = modelo.predict(img_procesada)
    
    clase_predicha = np.argmax(prediccion)  # Obtener índice de la clase con mayor probabilidad
    probabilidad = prediccion[0][clase_predicha]  # Obtener probabilidad de la clase predicha
    
    # Verificar si la confianza es menor al umbral
    if probabilidad < UMBRAL_CONF:
        resultado = "Sonido desconocido"
        probabilidad = 0.0
    else:
        resultado = f"{clases[clase_predicha]} ({probabilidad * 100:.2f}%)"
    
    # Mostrar la predicción en la terminal
    print(f"Predicción: {resultado}") 
    
    # Guardar la predicción en el archivo CSV
    with open(archivo_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([resultado, probabilidad * 100])

# Ruta de la imagen para predicción
ruta = "/home/camc/MisProyectos/Ejecucion/Bloque2/espectrograma.png"

# Especificar el archivo CSV para guardar las predicciones
archivo_csv = "/home/camc/MisProyectos/Ejecucion/predicciones.csv"

# Llamar a la función de predicción y guardar en CSV
predecir(ruta, archivo_csv)

print("¡¡¡ Predicción realizada !!!")