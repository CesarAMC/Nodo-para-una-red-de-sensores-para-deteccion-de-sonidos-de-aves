# Bibliotecas
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Carpetas de entrada y salida
audio_folder = "/home/camc/MisProyectos/wav_aves/ave2"
output_folder = "/home/camc/MisProyectos/espectros_aves/ave2"

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

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
        
        print(f"Espectrograma generado y convertido a RGB: {output_path}")
    except Exception as e:
        print(f"Error procesando {audio_path}: {e}")

# Procesar todos los archivos de la carpeta de audio
for file_name in os.listdir(audio_folder):
    if file_name.endswith(".wav"):
        audio_path = os.path.join(audio_folder, file_name)
        output_path = os.path.join(output_folder, file_name.replace(".wav", ".png"))
        print(f"Procesando {file_name}...")
        generate_spectrogram(audio_path, output_path)

print("¡Generación de espectrogramas en RGB completada!")