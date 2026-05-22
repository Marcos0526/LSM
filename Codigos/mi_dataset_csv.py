import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class MiDatasetLSM(Dataset):
    """
    Dataset personalizado para Lengua de Señas Mexicana (LSM).
    Lee archivos .csv organizados en carpetas por clase.
    Estructura esperada:
    root_dir/
        ├── hola/
        │   ├── muestra1.csv
        │   ├── muestra2.csv
        ├── gracias/
        │   ├── muestra1.csv
        ...
    """
    def __init__(self, root_dir, num_samples=32, num_nodes=75, num_dims=2):
        """
        Argumentos:
            root_dir (str): Ruta a la carpeta principal (ej. 'datos/train' o 'datos/val').
            num_samples (int): Número fijo de frames que el modelo leerá por video.
            num_nodes (int): Número de puntos clave (keypoints) extraídos por frame.
            num_dims (int): 2 si usaste (X, Y) o 3 si usaste (X, Y, Z).
        """
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_dims = num_dims

        self.archivos = []
        self.etiquetas = []
        
        # 1. Leer las carpetas para obtener las clases
        # Ignoramos archivos ocultos como .DS_Store
        carpetas = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes_ = sorted(carpetas)
        
        # 2. Crear un diccionario para convertir texto a número (ej. 'hola' -> 0)
        self.label_encoder = {nombre_clase: idx for idx, nombre_clase in enumerate(self.classes_)}

        # 3. Recorrer todas las carpetas y recolectar las rutas de los CSV
        for nombre_clase in self.classes_:
            ruta_clase = os.path.join(root_dir, nombre_clase)
            for nombre_archivo in os.listdir(ruta_clase):
                if nombre_archivo.endswith('.csv'):
                    self.archivos.append(os.path.join(ruta_clase, nombre_archivo))
                    self.etiquetas.append(self.label_encoder[nombre_clase])

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, idx):
        ruta_csv = self.archivos[idx]
        etiqueta_num = self.etiquetas[idx]

        # 1. Leer el CSV con Pandas
        # Asumimos que cada fila es un frame y las columnas son las coordenadas aplanadas
        df = pd.read_csv(ruta_csv)
        coordenadas = df.values

        # 2. Ajustar al número fijo de frames (num_samples)
        frames_actuales = coordenadas.shape[0]
        
        if frames_actuales < self.num_samples:
            # Si el video es muy corto, rellenamos copiando el último frame
            # (es mejor que poner ceros, ya que simula que la persona se quedó quieta)
            ultimo_frame = coordenadas[-1, :]
            padding = np.tile(ultimo_frame, (self.num_samples - frames_actuales, 1))
            coordenadas = np.vstack((coordenadas, padding))
        else:
            # Si el video es muy largo, lo recortamos a los primeros 'num_samples' frames
            coordenadas = coordenadas[:self.num_samples, :]

        # 3. Remodelar el tensor para el modelo GCN
        # Original shape: [num_samples, num_nodes * num_dims]
        # Transformación a: [num_nodes, num_samples * num_dims]
        
        # Primero dividimos los nodos y sus dimensiones (x,y)
        coordenadas = coordenadas.reshape(self.num_samples, self.num_nodes, self.num_dims)
        # Intercambiamos los ejes para que los nodos queden al principio: [nodos, frames, dimensiones]
        coordenadas = coordenadas.transpose(1, 0, 2) 
        # Volvemos a aplanar el final para que el GCN lo entienda: [nodos, frames * dimensiones]
        coordenadas = coordenadas.reshape(self.num_nodes, self.num_samples * self.num_dims)

        # 4. Convertir a tensores de PyTorch
        tensor_x = torch.FloatTensor(coordenadas)
        tensor_y = torch.tensor(etiqueta_num, dtype=torch.long)

        return tensor_x, tensor_y

    # Helper function para mantener compatibilidad con el script de entrenamiento
    class DummyEncoder:
        def __init__(self, classes):
            self.classes_ = classes

    @property
    def label_encoder_obj(self):
        """
        Retorna un objeto simulado (dummy) para que el script de entrenamiento
        pueda leer len(train_dataset.label_encoder.classes_) sin fallar.
        """
        return self.DummyEncoder(self.classes_)