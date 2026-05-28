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
- Marcos Bautista
- José Ramón García González
- Angel Ainar Lazcano Sánchez
- Victor A. Garmendia Fuentes
- Brenda Rafaela Mones Azcatl
- Rosario Saldaña de los Santos
---

Maestria en Investigacion en ciencia de Datos 
BUAP

## Instalación
El primer paso es tener un ambiente en nuestra computadora, qué es un ambiente te preguntarás, 
bueno imagina que en tu computadora tienes python 3.14, ahi todo corre muy bien
tienes todos tus archivos y todo funciona a la perfección, después de un año actualizas a python a 3.15, creas nuevos codigos usando python 3.15 y derrepente quieres correr un código viejo pero te encuentras con la sorpresa de que no funciona, que no es compatible, una solución es cambiarte nuevamente a python 3.14 pero ahora tus nuevos codigos que creaste con python 3.15 ya no funcionan.

La solución es crear un ambiente, los ambientes son como minicomputadoras dentro de tu computadora, imagina que tienes dos ambientes en uno instalamos python3.14 y en el otro 3.15, y en cada uno tenemos instalado numpy pero en uno esta la version de numpy que soporta python 3.14 y en el otro numpy 3.15, ahi el codigo funciona a la perfeccion.
Bueno, sólo hay que saber en que minicomputadora (ambiente) debemos de correr nuestro codigo.

Una herramienta que nos ayuda a crear estos ambiente es Conda, así que procederemos a instalarla en nuestra compu, todos los siguientes pasos se harán dentro de la terminal.

```
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

```
bash Miniconda3-latest-Linux-x86_64.sh
```

Aceptas licencia
Ruta por defecto (~/miniconda3)
Cuando pregunte: yes para inicializar

recuerda actualizar tu terminal, cerrarla y abrir una nueva también sirve

Ahora puedes clonar el repositorio en una carpeta o si ya lo hiciste no pasa nada

```
git clone git@github.com:Marcos0526/LSM.git
```
Ya tienes que tener SSH en tu computadora y git

Ahora verifica que tienes la versión más reciente del repositorio donde no aparece la carpeta LSM

ya en los archivos del repositorio clonado verás un archivo llamado environment.yml aquí estan todas las cosas que necesitamos
para poder correr el codigo, las paqueterias junto con sus versiones, procedemos a instalarlas

```
conda env create -f environment.yml
```
Por defecto el entorno usa CPU. Si tienes una tarjeta gráfica NVIDIA y quieres procesar el video en tiempo real más rápido, ejecuta este comando después de activar el entorno:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Para usar el env que acabamos de crear antes de correr el codigo debes de activarlo de la siguiente manera
```
conda activate LSM
```

Ahora ya puedes correr el codigo

Para desactivarlo basta con escribir
```
conda deactivate
```
Estas ready para empezar a usar el código.

Si algún día deseas desinstalar el ambeinte, porque ocupa espacio puedes hacer 
```
conda remove -n LSM --all
```
recuerda que siempre podrás recuperarlo gracias a environment.yml

cat << 'ENDOFFILE' >> README.md

---

## 🤖 Módulo LLM - Traducción de Glosas LSM al Español

Este módulo toma las glosas generadas por el modelo de visión y las traduce a oraciones en español natural usando un LLM via Groq API.

### 📁 Estructura de carpetas

    LLM/
    ├── simulacion_llm.py
    ├── outputs_vision/
    │   ├── video_01.txt
    │   └── ...
    └── traducciones/
        ├── video_01_traduccion.txt
        └── ...

### 📦 Requisitos

    pip install groq

### 🚀 Uso

1. Coloca los archivos .txt con glosas dentro de LLM/outputs_vision/
2. Ejecuta el script desde dentro de la carpeta LLM/:

    cd LLM
    python simulacion_llm.py

3. Ingresa tu API Key de Groq cuando se solicite (gsk_...)
4. Las traducciones se guardarán automáticamente en LLM/traducciones/

### 🔑 Obtener API Key

Regístrate en https://console.groq.com y genera una API Key gratuita.

### 🧠 Modelo utilizado

- llama-3.1-8b-instant via Groq API
ENDOFFILE
