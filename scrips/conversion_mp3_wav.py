# Bibliotecas
import os
from pydub import AudioSegment

# El programa cambia el formato mp3 a wav de los audios de las aves.
# Carpetas de entrada y salida
audio_folder = "/home/camc/MisProyectos/AudiosMP3/Ave_2"
output_folder = "/home/camc/MisProyectos/wav_aves/ave2"

# Función para convertir MP3 a WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    try:
        print(f"Convirtiendo {mp3_path} a WAV...")
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        print(f"Archivo convertido: {wav_path}")
    except Exception as e:
        print(f"Error al convertir {mp3_path}: {e}")

# Procesar todos los archivos de la carpeta de audio
for file_name in os.listdir(audio_folder):
    file_path = os.path.join(audio_folder, file_name)
    if file_name.endswith(".mp3"):
        wav_path = os.path.join(output_folder, file_name.replace(".mp3", ".wav"))
        convert_mp3_to_wav(file_path, wav_path)

print("¡Conversión de MP3 a WAV completada!")



