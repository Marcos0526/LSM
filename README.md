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

---
## Construcción de la base de datos

Para la construcción de la base de datos se realizó un proceso de grabación y registro de señas utilizando ambas manos: mano derecha y mano izquierda. La recopilación inició con el alfabeto, registrando las letras correspondientes mediante diferentes participantes. Dentro de este conjunto, algunas letras como **X, Q, K, Z, J y L** se consideran señas con movimiento, por lo que durante su captura fue necesario registrar no solo la posición final de la mano, sino también la trayectoria realizada durante la ejecución de la seña.

Además del alfabeto, también se comenzaron a registrar señas adicionales como **“hola”**, entre otras, con la finalidad de ampliar progresivamente la base de datos más allá de las letras individuales.

La captura de datos fue realizada con la participación de al menos siete personas, lo que permitió obtener variaciones naturales en la forma, posición, velocidad y ejecución de cada seña. Para cada letra o seña registrada se capturaron aproximadamente diez muestras por persona y por mano, generando un conjunto de datos más diverso y útil para el entrenamiento y evaluación del modelo.

El proceso de recopilación de datos se desarrolló utilizando **MediaPipe**, una herramienta de visión por computadora que permite detectar puntos clave del cuerpo humano. En este proyecto se utilizó para identificar y registrar referencias de las **manos, el rostro y el torso**, permitiendo capturar información relevante sobre la postura corporal, la ubicación de las manos y los movimientos realizados durante cada seña.

Las muestras recopiladas fueron organizadas en carpetas de acuerdo con la seña correspondiente y la mano utilizada. Esta estructura facilita la clasificación, el procesamiento posterior y el acceso ordenado a los datos, además de permitir que la base de datos sea escalable para futuras ampliaciones.

Para realizar nuevas grabaciones o ampliar la base de datos, el sistema cuenta con una opción integrada dentro del archivo **`Launcher.py`**. En dicho menú, la recopilación de nuevos datos se encuentra disponible en la **opción 3**, desde donde es posible registrar nuevas señas, agregar más muestras o incorporar nuevos participantes al conjunto de datos.

## Autor
- Marcos Bautista
- José Ramón García González
- Angel Ainar Lazcano Sánchez
- Victor A. Garmendia Fuentes
- Brenda Rafaela Mones Azcatl
- Rosario Saldaña de los Santos
- Ailín Atala Beltrán Guzmán
- Erick Michel Meza Mendoza
- Raúl Badillo Lora
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
### 📄 Arquitectura y Referencias

En este proyecto se implementó la arquitectura de red TGCN (Temporal Graph Convolutional Network) utilizando PyTorch, basándose en el trabajo de Li et al. Toda la definición y lógica principal de esta red neuronal se encuentra encapsulada en el archivo `tgcn_model.py`. El modelo interpreta el conjunto de coordenadas de entrada considerando el cuerpo humano y las manos como un grafo, donde los nodos son los landmarks extraídos en los fotogramas continuos.

**Flujo y Procesamiento (TGCN):**
El flujo de los datos dentro de `tgcn_model.py` comienza con una capa inicial de convolución en grafos con atención (`GraphConvolution_att`), seguida de un apilamiento de bloques residuales (`GC_Block`). A diferencia de una GCN clásica, esta arquitectura implementa una matriz de atención paramétrica y entrenable (`self.att`) que aprende ponderaciones dinámicas entre los nodos, capturando las relaciones complejas del movimiento sin depender exclusivamente de las conexiones físicas predefinidas. 

Cada bloque residual consta de dos capas de convolución, normalización por lotes (*Batch Normalization*), funciones de activación Tanh y *Dropout* para regularización, una conexión residual (`y + x`) que facilita el flujo del gradiente en capas profundas.

**Data Augmentation:**
Para garantizar que el modelo generalice de manera robusta y evitar el sobreajuste (*overfitting*), aplicamos técnicas de aumento de datos directamente sobre los tensores. Esta lógica de transformación se encuentra implementada en el archivo `mi_dataset_csv.py`. Durante el proceso, cada secuencia temporal tiene un 50% de probabilidad independiente de sufrir estas modificaciones:
* **Rotación aleatoria en 3D (eje Y):** Ángulos aleatorios entre -25 y 25 grados para simular cambios de perspectiva de la cámara.
* **Escalado espacial:** Modificación de la escala entre un 80% y 120% para compensar distancias variables del sujeto.
* **Ruido Gaussiano:** Inyección de ruido sutil (media 0, desviación 0.003) para emular la incertidumbre y el *jitter* natural de la extracción de landmarks.

**Salida y Entrenamiento:**
Tras analizar la evolución temporal a través de todos los bloques, la red aplica un agrupamiento global (*Global Average Pooling*) mediante `torch.mean`, colapsando las dimensiones del tensor. Posteriormente pasa por un *Dropout* final y una capa densa (Fully Connected) que proyecta las características a la predicción final de la clase o seña detectada.

Para iniciar el proceso de entrenamiento uniendo el dataset aumentado y la arquitectura descrita, simplemente se debe ejecutar:

```bash
python train_tgcn.py
```

**Cita del Artículo:**
Si desean revisar la fundamentación matemática en la que nos basamos para el uso de convoluciones en secuencias de poses, pueden consultar el trabajo original aquí:

```bibtex
@inproceedings{li2020transferring,
  title={Transferring cross-domain knowledge for video sign language recognition},
  author={Li, Dongxu and Yu, Xin and Xu, Chenchen and Petersson, Lars and Li, Hongdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6205--6214},
  year={2020}
}
```

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
