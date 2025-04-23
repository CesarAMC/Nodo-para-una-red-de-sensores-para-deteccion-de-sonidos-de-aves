# Desarrollo de un nodo para una red de sensores para la deteccion de sonidos de aves mexicanas

Este proyecto hace uso de redes neuronales para el reconocimiento de aves mediante la imagen
del espectro de su canto. El sistema consta de 3 bloques, los cuales son:
1- Bloque de adquisicion de audio.
2- Bloque de procesamiento.
3- Bloque de desplegado.

En el primer bloque se obtiene el audio, esto mediante la tecnica de detecciòn por umbral,
el sistema se mantiene escuchando continuamente, cuamdo un sonido supera el umbral previa-
mente definido, se graba el sonido durante 10 segundos, este audio es guardado en un archi-
vo .wav.

El segundo bloque lo que hace es el procesamiento del audio previamente obtenido en el pri-
mer bloque,lo que hace es obtener la imagen del espectro del audio. Esta imagen es utilizada
por una CNN previamente entrenada para realizar una prediccion.

El tercer bloque lo que hace es mostrar el resultado de la prediccion en pantalla al usuario,
ademas cada prediccion es guardada en un archivo CSV.

## Estructura del proyecto

- 'scrips/': Scrips principales en Python.
- 'wav_aves': Archivos de audio de sonidos de aves (No se incluye).
- 'wav_aves/ave1':Audios ave1.
- 'wav_aves/ave2':Audios ave2.
- 'espectros_aves': Espectros usados para entrenar los modelos CNN.
- 'espectros_aves/ave1': Imagenes espectros ave1.
- 'espectros_aves/ave2': Imagenes espectros ave2.
- 'Ejecucion': Carpeta donde se guardan los archivos crados al correr el programa, incluido el archivo CSV (No se incluye).
- 'Ejecucion/Bloque1': Se guarda el audio grabado de 10 seg de duracion.
- 'Ejecucion/Bloque2': Se guarda la imagen del espectro del audio grabado.
- 'modelo_CNNaves_MobileNetV2.keras': Modelo CNN entrenado.
- 'myPT2', 'PryPrm2_CNN': Entornos virtuyales (No se incluyen).

## Requisitos
Si deseas hacer la conversion de los archivos de audio de mp3 a wav, debes descargarlos de la biblioteca virtual de aves mexicanas
Xeno-Canto:
1. Ave1: https://xeno-canto.org/set/3518?filter=trogon+mexicanus
2. Ave2: https://xeno-canto.org/set/3518?filter=Arremon
 
Consulta los isguientes archivos para conocer las dependencias necesarias:

- 'requirements_tf.txt': Entorno myPT2.
- 'requirements_other.txt': Entorno PryTrm2_CNN.

## Como ejecutar
### Pasos para entrenar la CNN:
- NOTA: Si no deseas descargar los audios utiliza la carpeta de los espectros para el entrenamiento, pasa al paso 4.
1. Tener los audios descargados, son 100 por clase, se utilizo la biblioteca digital Xeno-Canto.
2. Ejecutar el script para la conversion mp3 a wav: conversion_mp3_wav.py
3. Ejecutar el script EspectrosRGB.py para obtener las imagenes con las que se hace el entrenamiento.
4. Ejecuta el script CNN_MovileNetV2.py para entrenar la red.

### Pasos para ejecutar el sistema completo:
1. Actibar el entorno virtual:
	source myPT2/bin/activate

2. Ejecuta el script principal:
	python scripts/Bloque_Adquisicion_Audio.py 

NOTA: ASEGURATE DE TENER LA ESTRUCTURA CORRECTA DEL PROYECTO DE LO CONTRARIO REVISA LAS RUTAS DE LOS SCRIPTS
QUE INICIAN CON Bloque y Prediccion.
