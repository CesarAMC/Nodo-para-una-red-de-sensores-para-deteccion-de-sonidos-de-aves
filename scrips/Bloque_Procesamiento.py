# Bibliotecas
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# El codigo de este programa lo que hace es procesar el audio grabado
# para obtener la imagen del espectro del audio para realizar una prediccion 
# con la CNN previamente entrenada y asi saber si es una de las 4 aves con las que
# se entreno la CNN.

# Ruta de entrada y salida
audio_path = "/home/camc/MisProyectos/Ejecucion/Bloque1/grabacion.wav"
output_path = "/home/camc/MisProyectos/Ejecucion/Bloque2/espectrograma.png"

# Asegurar que la carpeta de salida exista
# os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Función para generar el espectrograma y convertirlo a RGB
def generate_spectrogram(audio_path, output_path, img_size=(224, 224)):
    try:
        # Cargar el archivo de audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calcular el espectrograma
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # Crear figura para el espectrograma
        fig, ax = plt.subplots(figsize=(img_size[0] / 100, img_size[1] / 100), dpi=100)
        librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log', cmap='viridis', ax=ax)
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Guardar como imagen temporal
        temp_output = output_path.replace(".png", "_temp.png")
        plt.savefig(temp_output, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # Convertir a RGB
        with Image.open(temp_output) as img:
            img_rgb = img.convert("RGB")
            img_rgb.save(output_path)
        
        # Eliminar imagen temporal
        os.remove(temp_output)
        
       # print(f"Espectrograma generado y convertido a RGB: {output_path}")
    except Exception as e:
        print(f"Error procesando {audio_path}: {e}")

# Procesar el único archivo de audio
generate_spectrogram(audio_path, output_path)

print("¡Generación de espectrograma completada!")