#Bibliotecas
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
import wave
import time
import subprocess

# Este codigo lo que hace es grabar 10 segundos de audio
# cuando se llega al umbral de activacion y se graban solo
# las frecuencias de 1-7kHz de acuerdo al filtro pasabandas. 
# Parámetros.
SAMPLE_RATE = 44100  # Frecuencia de muestreo (Hz)
DURATION = 10  # Segundos de grabación
LOWCUT = 1000  # f_min (kHz)
HIGHCUT = 7000  # f_max (kHz)
THRESHOLD = 0.02  # Umbral de activación

# Ruta programas Bloque 2 y 3.
procesamiento = "/home/camc/MisProyectos/scrips/Bloque_Procesamiento.py"
prediccion = "/home/camc/MisProyectos/scrips/PrediccionCNN.py"

# Filtro pasa banda.
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Función para aplicar el filtro.
def filtro_pasabanda(audio, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, audio)

# Función para grabar (10s).
def grabar_audio():
    print("Grabando...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    
    # Guardar audio (wav)
    with wave.open("/home/camc/MisProyectos/Ejecucion/Bloque1/grabacion.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    
    print("Guardado como 'grabacion.wav'")

# Mensaje de inicio del bucle
print("Iniciando porgrama:")
print("Esperando sonido en el rango", LOWCUT, "-", HIGHCUT, "Hz...")

while True:

    # Capturar 0.5 segundos de audio para análisis
    audio_chunk = sd.rec(int(0.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()

    # Aplicar filtro pasa banda
    audio_filtrado = filtro_pasabanda(audio_chunk[:, 0], LOWCUT, HIGHCUT, SAMPLE_RATE)

    # Calcular RMS del audio filtrado
    rms = np.sqrt(np.mean(np.square(audio_filtrado)))

    # Mostrar nivel de sonido en la terminal
    print(f"Nivel de sonido (RMS): {rms:.4f}")

    # Activar grabación si el RMS supera el umbral
    if rms > THRESHOLD:
        print("Sonido detectado en el rango de frecuencias.")
        grabar_audio()

        # Ejecutar el procesamiento del audio
        print("Procesando el audio grabado...")
        subprocess.run(["/home/camc/MisProyectos/PryTrm2_CNN/bin/python", procesamiento])

        # Ejecutar la predicción con la CNN
        print("Realizando predicción con la CNN...")
        subprocess.run(["/home/camc/MisProyectos/myPT2/bin/python", prediccion])
        print("Proceso completado. Volviendo a la espera de otro sonido...\n")
        time.sleep(5)
    
    # Agregar un pequeño retardo para evitar saturar el CPU
    time.sleep(0.5)