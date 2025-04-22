# Desarrollo de un nodo para una red de sensores para la deteccion de sonidos de aves mexicanas

Este proyecto hace uso de redes neuronales para el reconocimiento de aves mediante la imagen
del espectro de su canto. El sistema consta de 3 bloques, los cuales son:
1- Bloque de adquisicion de audio.
2- Bloque de procesamiento.
3- Bloque de desplegado.

En el primer bloque se obtiene el audio, esto mediante la tecnica de detecciòn por umbral,
el sistema se mantiene escuchando continuamente, cuamdo un sonido supera el umbral previa-
mente definido, se graba el sonido durante 10 segundos, este audio es guardado en un archi-
vo .wav..
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
- 'espectros_aves': Espectros usados para entrenar los modelos CNN (No se incluyen).
- 'espectros_aves/ave1': Imagenes espectros ave1.
- 'espectros_aves/ave2': Imagenes espectros ave2.
- 'Ejecucion': Carpeta donde se guardan los archivos crados al correr el programa, incluido el archivo CSV (No se incluye).
- 'Ejecucion/Bloque1': Se guarda el audio grabado de 10 seg de duracion.
- 'Ejecucion/Bloque2': Se guarda la imagen del espectro del audio grabado.
- 'modelo_CNNaves_MobileNetV2.keras': Modelo CNN entrenado.
- 'myPT2', 'PryPrm2_CNN': Entornos virtuyales (No se incluyen).

## Requisitos

Consulta los isguientes archivos para conocer las dependencias necesarias:

- 'requirements_tf.txt': Entorno myPT2.
- 'requirements_other.txt': Entorno PryTrm2_CNN.

## Como ejecutar

1. Actibar el entorno virtual:
	source myPT2/bin/activate

2. Ejecuta el script principal:
	python scripts/Bloque_Adquisicion_Audio.py 

NOTA: ASEGURATE DE TENER LA ESTRUCTURA CORRECTA DEL PROYECTO DE LO CONTRARIO REVISA LAS RUTAS DE LOS SCRIPTS
QUE INICIAN CON Bloque y Prediccion.
