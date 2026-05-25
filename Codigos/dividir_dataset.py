import os
import shutil
import random

def split_dataset(origen_dir, destino_dir, train_ratio=0.9):
    """
    Divide un dataset organizado en carpetas (origen_dir/clase/*.npy)
    en conjuntos de train y val dentro de destino_dir.
    """
    # 1. Crear carpetas principales
    train_dir = os.path.join(destino_dir, 'train')
    val_dir = os.path.join(destino_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 2. Obtener todas las carpetas de clases (a, b, hola, etc.)
    # Ignoramos archivos sueltos, solo buscamos directorios
    clases = [d for d in os.listdir(origen_dir) if os.path.isdir(os.path.join(origen_dir, d))]
    
    if not clases:
        print(f"Error: No se encontraron carpetas de clases en {origen_dir}")
        return

    print(f"Clases detectadas: {clases}\n")
    
    total_train = 0
    total_val = 0
    
    # 3. Procesar cada clase individualmente
    for clase in clases:
        ruta_clase_origen = os.path.join(origen_dir, clase)
        
        # Obtener todos los archivos .npy de esta clase específica
        archivos = [f for f in os.listdir(ruta_clase_origen) if f.endswith('.npy')]
        
        if not archivos:
            continue
            
        # Mezclar aleatoriamente para evitar sesgos de tiempo 
        # (ej. si al principio hacías la seña diferente que al final)
        random.seed(42) # Semilla fija para reproducibilidad (opcional)
        random.shuffle(archivos)
        
        # Calcular el punto exacto de corte (90%)
        corte = int(len(archivos) * train_ratio)
        
        train_archivos = archivos[:corte]
        val_archivos = archivos[corte:]
        
        # Crear las subcarpetas de esta clase en train y val
        ruta_clase_train = os.path.join(train_dir, clase)
        ruta_clase_val = os.path.join(val_dir, clase)
        
        os.makedirs(ruta_clase_train, exist_ok=True)
        os.makedirs(ruta_clase_val, exist_ok=True)
        
        # Copiar archivos al conjunto de entrenamiento
        for f in train_archivos:
            # shutil.copy2 preserva los metadatos originales del archivo (fecha de creación)
            shutil.copy2(os.path.join(ruta_clase_origen, f), os.path.join(ruta_clase_train, f))
            
        # Copiar archivos al conjunto de validación
        for f in val_archivos:
            shutil.copy2(os.path.join(ruta_clase_origen, f), os.path.join(ruta_clase_val, f))
            
        print(f"Clase '{clase}': {len(train_archivos)} a train, {len(val_archivos)} a val.")
        
        total_train += len(train_archivos)
        total_val += len(val_archivos)

    # 4. Reporte final
    print("\n" + "="*30)
    print("RESUMEN DE LA DIVISIÓN")
    print("="*30)
    print(f"Total de archivos en Train: {total_train}")
    print(f"Total de archivos en Val:   {total_val}")
    print("¡Proceso completado con éxito!")

if __name__ == "__main__":
    # Rutas relativas basadas en la estructura de tu proyecto
    DIRECTORIO_ORIGEN = "../Dataset/sequences" 
    DIRECTORIO_DESTINO = "../dataset_procesado"
    
    # train_ratio=0.9 indica 90% train, 10% validación
    split_dataset(DIRECTORIO_ORIGEN, DIRECTORIO_DESTINO, train_ratio=0.90)