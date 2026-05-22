import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

import utils
from configs import Config
from tgcn_model import GCN_muti_att
from mi_dataset_csv import MiDatasetLSM  # <-- Importamos TU nuevo dataset
from train_utils import train, validation

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run(pose_data_root, configs, save_model_to=None):
    epochs = configs.max_epochs
    log_interval = configs.log_interval
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages

    # --- PARÁMETROS CLAVE PARA TUS CSVs ---
    n_nodes = 75  # Número de puntos clave (ajusta según tu extracción)
    n_dims = 2    # 2 para (X,Y) o 3 para (X,Y,Z)

    # setup dataset (Entrenamiento)
    # Asume que dentro de pose_data_root hay una carpeta llamada 'train'
    train_dataset = MiDatasetLSM(root_dir=os.path.join(pose_data_root, 'train'), 
                                 num_samples=num_samples, 
                                 num_nodes=n_nodes, 
                                 num_dims=n_dims)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    batch_size=configs.batch_size,
                                                    shuffle=True)

    # setup dataset (Validación)
    # Asume que dentro de pose_data_root hay una carpeta llamada 'val'
    val_dataset = MiDatasetLSM(root_dir=os.path.join(pose_data_root, 'val'), 
                               num_samples=num_samples, 
                               num_nodes=n_nodes, 
                               num_dims=n_dims)

    # shuffle=False es una buena práctica para el set de validación
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                  batch_size=configs.batch_size,
                                                  shuffle=False) 

    # Imprimir las clases detectadas de las carpetas
    logging.info('\n'.join(['Class labels are: '] + [(str(i) + ' - ' + label) for i, label in
                                                     enumerate(train_dataset.classes_)]))

    # setup the model
    model = GCN_muti_att(input_feature=num_samples * n_dims, 
                         hidden_feature=hidden_size,  # Usa el tamaño oculto de la config
                         num_class=len(train_dataset.classes_), 
                         p_dropout=drop_p, 
                         num_stage=num_stages,
                         num_nodes=n_nodes).cuda()  # <-- Coma corregida y num_nodes añadido

    # setup training parameters, learning rate, optimizer, scheduler
    lr = configs.init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_val_losses = []
    epoch_val_scores = []

    best_test_acc = 0
    best_epoch_num = 0

    # Crear directorios de salida para evitar errores de guardado
    os.makedirs('output', exist_ok=True)
    os.makedirs(os.path.join('checkpoints', subset), exist_ok=True)

    # start training
    for epoch in range(int(epochs)):
        print('start training.')
        train_losses, train_scores, train_gts, train_preds = train(log_interval, model,
                                                                   train_data_loader, optimizer, epoch)
        print('start testing.')
        val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
                                                                                val_data_loader, epoch,
                                                                                save_to=save_model_to)

        logging.info('========================\nEpoch: {} Average loss: {:.4f}'.format(epoch, val_loss))
        logging.info('Top-1 acc: {:.4f}'.format(100 * val_score[0]))
        logging.info('Top-3 acc: {:.4f}'.format(100 * val_score[1]))
        logging.info('Top-5 acc: {:.4f}'.format(100 * val_score[2]))
        logging.info('Top-10 acc: {:.4f}'.format(100 * val_score[3]))
        logging.info('Top-30 acc: {:.4f}'.format(100 * val_score[4]))
        logging.debug('mislabelled val. instances: ' + str(incorrect_samples))

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_val_losses.append(val_loss)
        epoch_val_scores.append(val_score[0])

        # save all train test results
        np.save('output/epoch_training_losses.npy', np.array(epoch_train_losses))
        np.save('output/epoch_training_scores.npy', np.array(epoch_train_scores))
        np.save('output/epoch_test_loss.npy', np.array(epoch_val_losses))
        np.save('output/epoch_test_score.npy', np.array(epoch_val_scores))

        if val_score[0] > best_test_acc:
            best_test_acc = val_score[0]
            best_epoch_num = epoch

            torch.save(model.state_dict(), os.path.join('checkpoints', subset, 'gcn_epoch={}_val_acc={}.pth'.format(
                best_epoch_num, best_test_acc)))

    # Graficar y guardar matrices de confusión
    class_names = train_dataset.classes_
    utils.plot_confusion_matrix(train_gts, train_preds, classes=class_names, normalize=False,
                                save_to='output/train-conf-mat')
    utils.plot_confusion_matrix(val_gts, val_preds, classes=class_names, normalize=False, save_to='output/val-conf-mat')


if __name__ == "__main__":
    # 1. 'directorio_actual' detectará que estamos dentro de la carpeta "Codigos"
    directorio_actual = os.path.dirname(os.path.abspath(__file__))

    # 2. 'proyecto_raiz' sube un nivel para llegar a la carpeta principal (donde está README.md)
    proyecto_raiz = os.path.dirname(directorio_actual)

    subset = 'asl100'

    # 3. Navegamos hacia la carpeta del dataset. 
    # Asumo que tus carpetas 'train/' y 'val/' están dentro de 'Dataset/sequences/'
    pose_data_root = os.path.join(proyecto_raiz, 'Dataset', 'sequences') 
    
    # 4. El archivo de configuración sigue estando dentro de Codigos/configs/
    config_file = os.path.join(directorio_actual, 'configs', '{}.ini'.format(subset))
    configs = Config(config_file)

    # Crear carpeta de salida dentro de 'Codigos' para guardar los logs y pesos
    os.makedirs(os.path.join(directorio_actual, 'output'), exist_ok=True)
    os.makedirs(os.path.join(directorio_actual, 'checkpoints', subset), exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(directorio_actual, 'output', '{}.log'.format(os.path.basename(config_file)[:-4])), 
        level=logging.DEBUG, 
        filemode='w+'
    )

    logging.info('Calling main.run()')
    # Ejecutamos el entrenamiento
    run(pose_data_root=pose_data_root, configs=configs)
    logging.info('Finished main.run()')