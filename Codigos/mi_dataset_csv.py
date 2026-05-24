import os
import torch
import numpy as np
from torch.utils.data import Dataset

class MiDatasetLSM(Dataset):
    def __init__(self, root_dir, num_samples=32, num_nodes=75, num_dims=2):
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_dims = num_dims

        self.archivos = []
        self.etiquetas = []
        
        # 1. Obtener todos los archivos .npy en la carpeta (ej. 'Dataset/train/')
        archivos_en_carpeta = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        
        # 2. Identificar las clases únicas basándose en la primera letra
        # Ej: si hay 'j_...' y 'k_...', las clases son ['j', 'k']
        clases_unicas = sorted(list(set([f[0].lower() for f in archivos_en_carpeta])))
        self.classes_ = clases_unicas
        self.label_encoder = {char: idx for idx, char in enumerate(clases_unicas)}

        # 3. Guardar rutas y etiquetas
        for nombre_archivo in archivos_en_carpeta:
            self.archivos.append(os.path.join(root_dir, nombre_archivo))
            inicial = nombre_archivo[0].lower()
            self.etiquetas.append(self.label_encoder[inicial])

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, idx):
        ruta_npy = self.archivos[idx]
        etiqueta_num = self.etiquetas[idx]
        
        # 1. Obtener el nombre del archivo para usarlo como ID
        nombre_id = os.path.basename(ruta_npy)

        # 2. Cargar el archivo .npy
        coordenadas = np.load(ruta_npy)

        # 3. Ajustar al número fijo de frames
        frames_actuales = coordenadas.shape[0]
        if frames_actuales < self.num_samples:
            ultimo_frame = coordenadas[-1, :, :]
            padding = np.tile(ultimo_frame, (self.num_samples - frames_actuales, 1, 1))
            coordenadas = np.vstack((coordenadas, padding))
        else:
            coordenadas = coordenadas[:self.num_samples, :, :]

        # 4. Reshape para GCN
        coordenadas = coordenadas.transpose(1, 0, 2) 
        coordenadas = coordenadas.reshape(self.num_nodes, self.num_samples * self.num_dims)

        # 5. Retornar los TRES valores que tu train_utils espera
        return torch.FloatTensor(coordenadas), torch.tensor(etiqueta_num, dtype=torch.long), nombre_id