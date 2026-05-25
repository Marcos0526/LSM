import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset

import utils
from configs import Config
from tgcn_model import GCN_muti_att
from mi_dataset_csv import MiDatasetLSM 
from train_utils import train, validation

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run(pose_data_root, configs):
    epochs = configs.max_epochs
    log_interval = configs.log_interval
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages

    # --- PARÁMETROS CLAVE ---
    n_nodes = 79
    n_dims = 3    

    # setup dataset (Entrenamiento)
    train_dataset = MiDatasetLSM(root_dir=os.path.join(pose_data_root, 'train'), 
                                 num_samples=num_samples, 
                                 num_nodes=n_nodes, 
                                 num_dims=n_dims,
                                 num_variations= 20)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    batch_size=configs.batch_size,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    pin_memory=True)

    # setup dataset (Validación)
    val_dataset = MiDatasetLSM(root_dir=os.path.join(pose_data_root, 'val'), 
                               num_samples=num_samples, 
                               num_nodes=n_nodes, 
                               num_dims=n_dims,
                               num_variations= 50)

    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                  batch_size=configs.batch_size,
                                                  shuffle=False) 

    # Imprimir las clases detectadas
    logging.info('\n'.join(['Class labels are: '] + [(str(i) + ' - ' + label) for i, label in
                                                     enumerate(train_dataset.classes_)]))

    # setup the model
    model = GCN_muti_att(input_feature=num_samples * n_dims, 
                         hidden_feature=hidden_size,
                         num_class=len(train_dataset.classes_), 
                         p_dropout=drop_p, 
                         num_stage=num_stages,
                         num_nodes=n_nodes).cuda()

    # setup training parameters
    lr = configs.init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)

    epoch_train_losses = []
    epoch_train_scores = []
    epoch_val_losses = []
    epoch_val_scores = []

    best_test_acc = 0
    
    # Crear directorios
    os.makedirs('output', exist_ok=True)
    checkpoint_dir = os.path.join('checkpoints', subset)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # start training
    for epoch in range(int(epochs)):
        print(f'\n--- Epoch: {epoch + 1} ---')
        print('start training.')
        train_losses, train_scores, train_gts, train_preds = train(log_interval, model,
                                                                   train_data_loader, optimizer, epoch)
        print('start testing.')
        # Pasamos None a save_to porque controlaremos el guardado manualmente
        val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
                                                                                val_data_loader, epoch,
                                                                                save_to=None)

        logging.info('========================\nEpoch: {} Average loss: {:.4f}'.format(epoch, val_loss))
        logging.info('Top-1 acc: {:.4f}'.format(100 * val_score[0]))
        logging.debug('mislabelled val. instances: ' + str(incorrect_samples))

        # Guardar resultados históricos
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_val_losses.append(val_loss)
        epoch_val_scores.append(val_score[0])

        np.save('output/epoch_training_losses.npy', np.array(epoch_train_losses))
        np.save('output/epoch_test_score.npy', np.array(epoch_val_scores))

        # GUARDADO DEL MEJOR MODELO
        if val_score[0] > best_test_acc:
            best_test_acc = val_score[0]
            ruta_best_model = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), ruta_best_model)
            print(f"✅ ¡Nuevo mejor modelo guardado con precisión {best_test_acc:.4f}!")

    # Graficar matrices de confusión
    class_names = train_dataset.classes_
    utils.plot_confusion_matrix(train_gts, train_preds, classes=class_names, normalize=False,
                                save_to='output/train-conf-mat')
    utils.plot_confusion_matrix(val_gts, val_preds, classes=class_names, normalize=False, save_to='output/val-conf-mat')


if __name__ == "__main__":
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    proyecto_raiz = os.path.dirname(directorio_actual)
    subset = 'asl100'

    pose_data_root = os.path.join(proyecto_raiz, 'Codigos/Dataset')
    config_file = os.path.join(directorio_actual, 'configs', '{}.ini'.format(subset))
    configs = Config(config_file)

    os.makedirs(os.path.join(directorio_actual, 'output'), exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(directorio_actual, 'output', '{}.log'.format(os.path.basename(config_file)[:-4])), 
        level=logging.DEBUG, 
        filemode='w+'
    )

    logging.info('Calling main.run()')
    run(pose_data_root=pose_data_root, configs=configs)
    logging.info('Finished main.run()')