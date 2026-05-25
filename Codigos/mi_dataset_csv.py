import os
import torch
import numpy as np
from torch.utils.data import Dataset

class MiDatasetLSM(Dataset):
    def __init__(self, root_dir, num_samples=120, num_nodes=79, num_dims=3, is_train=True, num_variations=5):
        """
        is_train: Si es True, multiplica el dataset usando Data Augmentation.
                  Si es False (validación), usa solo los datos reales.
        """
        self.root_dir = root_dir
        self.num_samples = num_samples # Actualizado a 120
        self.num_nodes = num_nodes     # Actualizado a la topología de 79
        self.num_dims = num_dims       # Actualizado a 3D (x, y, z)
        self.is_train = is_train

        # 1. Obtener todos los archivos .npy
        archivos_en_carpeta = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        
        # 2. Identificar las clases (Toma la primera letra)
        clases_unicas = sorted(list(set([f[0].lower() for f in archivos_en_carpeta])))
        self.classes_ = clases_unicas
        self.label_encoder = {char: idx for idx, char in enumerate(clases_unicas)}

        # 3. CONSTRUCCIÓN DEL ÍNDICE VIRTUAL
        # Aquí guardamos las "instrucciones" de qué cargar, en lugar del archivo pesado en RAM
        self.muestras_virtuales = []
        
        for nombre_archivo in archivos_en_carpeta:
            ruta_npy = os.path.join(root_dir, nombre_archivo)
            inicial = nombre_archivo[0].lower()
            etiqueta = self.label_encoder[inicial]
            
            # Siempre agregamos la muestra original real
            self.muestras_virtuales.append({
                'ruta': ruta_npy,
                'etiqueta': etiqueta,
                'es_aumentada': False,
                'aug_id': 0
            })
            
            # Si estamos entrenando, generamos los "punteros" para las copias aumentadas
            if self.is_train:
                for i in range(num_variations):
                    self.muestras_virtuales.append({
                        'ruta': ruta_npy,
                        'etiqueta': etiqueta,
                        'es_aumentada': True,
                        'aug_id': i + 1 # ID para distinguirlas (aug1, aug2...)
                    })

    def __len__(self):
        # Ahora el DataLoader creerá que el dataset es (originales * (1 + variaciones)) veces más grande
        return len(self.muestras_virtuales)

    def aplicar_data_augmentation(self, secuencia):
        """Aplica UNA variación aleatoria a un solo tensor"""
        seq_aug = np.copy(secuencia)
        
        # 1. Rotación aleatoria en Y
        theta = np.radians(np.random.uniform(-25, 25))
        c, s = np.cos(theta), np.sin(theta)
        Ry = np.array([[c, 0, s], 
                       [0, 1, 0], 
                       [-s, 0, c]])
        seq_aug = np.dot(seq_aug, Ry.T)
        
        # 2. Escalado espacial
        scale_factor = np.random.uniform(0.80, 1.20)
        seq_aug = seq_aug * scale_factor
        
        # 3. Ruido Gaussiano
        noise = np.random.normal(0.001, 0.004, seq_aug.shape)
        seq_aug = seq_aug + noise
        
        return seq_aug

    def __getitem__(self, idx):
        # 1. Consultar el índice virtual para ver qué nos toca generar
        item = self.muestras_virtuales[idx]
        ruta_npy = item['ruta']
        etiqueta_num = item['etiqueta']
        
        nombre_id = os.path.basename(ruta_npy)

        # 2. Cargar el archivo .npy original
        coordenadas = np.load(ruta_npy)

        # 3. Transformación "Al vuelo": Solo si el índice marca que es un clon
        if item['es_aumentada']:
            coordenadas = self.aplicar_data_augmentation(coordenadas)
            # Modificamos el nombre_id para evitar colisiones en tus métricas (ej. aug3_j_123.npy)
            nombre_id = f"aug{item['aug_id']}_{nombre_id}"

        # 4. Ajustar al número fijo de frames (Relleno / Truncado)
        frames_actuales = coordenadas.shape[0]
        if frames_actuales < self.num_samples:
            ultimo_frame = coordenadas[-1, :, :]
            padding = np.tile(ultimo_frame, (self.num_samples - frames_actuales, 1, 1))
            coordenadas = np.vstack((coordenadas, padding))
        else:
            coordenadas = coordenadas[:self.num_samples, :, :]

        # 5. Reshape estricto para arquitectura GCN
        # Original: (frames, nodos, dims) -> Transpose: (nodos, frames, dims)
        coordenadas = coordenadas.transpose(1, 0, 2) 
        coordenadas = coordenadas.reshape(self.num_nodes, self.num_samples * self.num_dims)

        # 6. Retornar los tensores a PyTorch
        return torch.FloatTensor(coordenadas), torch.tensor(etiqueta_num, dtype=torch.long), nombre_id