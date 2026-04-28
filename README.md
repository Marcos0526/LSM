# 🖐️ Transcripción en Tiempo Real de Lengua de Señas Mexicana (LSM)

Sistema de visión por computadora y aprendizaje automático para la **detección, reconocimiento y transcripción en tiempo real de Lengua de Señas Mexicana (LSM)** utilizando **OpenCV** y **MediaPipe**.

---

## 📌 Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un sistema capaz de:

- Detectar manos en tiempo real mediante cámara web.
- Extraer puntos clave (landmarks) usando MediaPipe.
- Procesar secuencias de movimientos.
- Clasificar señas del alfabeto y palabras en LSM.
- Generar transcripción automática en texto.

El propósito es contribuir a la **accesibilidad tecnológica** y facilitar la comunicación entre personas sordas y oyentes mediante herramientas basadas en IA.

---

## ⚙️ Arquitectura del Sistema

Captura de Video (Webcam)
        ↓
Detección de Mano (MediaPipe)
        ↓
Extracción de Landmarks (21 puntos)
        ↓
Preprocesamiento de Datos
        ↓
Modelo de Clasificación
        ↓
Salida en Texto (Transcripción)



## Autor
Marcos Bautista
José Ramón García González

(Pomgan sus nombres)

---

Maestria en Investigacion en ciencia de Datos 
BUAP

## Instalación
Para empezar clona el repositorio en una carpeta

```
git clone git@github.com:Marcos0526/LSM.git
```
Ya tienes que tener SSH en tu computadora y git

Ahora verifica que tienes la versión más reciente del repositorio donde no aparece la carpeta LSM
como segundo paso en la terminal en la carpeta del repositorio ejecuta
```
python3  -m venv .LSM
```

despues podrás instalar las paqueterías necesarias

```
pip install -r requirements.txt
```
Estas ready para empezar a usar el código.
