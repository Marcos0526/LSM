import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def train(log_interval, model, train_loader, optimizer, epoch):
    # set model as training mode
    model.train() # AÑADIDO: Es una buena práctica asegurar que el modelo está en modo train
    losses = []
    scores = []
    train_labels = []
    train_preds = []

    N_count = 0  # counting total trained sample in one epoch
    for batch_idx, data in enumerate(train_loader):
        X, y, video_ids = data
        # distribute data to device
        X, y = X.cuda(), y.cuda().view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        out = model(X)  # output has dim = (batch, number of classes)

        loss = compute_loss(out, y)

        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(out, 1)[1]  # y_pred != output

        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())

        # collect prediction labels
        train_labels.extend(y.cpu().data.squeeze().tolist())
        train_preds.extend(y_pred.cpu().data.squeeze().tolist())

        scores.append(step_score)  # computed on CPU

        loss.backward()

        optimizer.step()

        # --- CORRECCIÓN: Silenciamos este print para evitar el spam en la consola ---
        # if (batch_idx + 1) % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.6f}%'.format(
        #         epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
        #         100 * step_score))

    return losses, scores, train_labels, train_preds


def validation(model, test_loader, epoch, save_to):
    # set model as testing mode
    model.eval()

    val_loss = []
    all_y = []
    all_y_pred = []
    all_video_ids = []
    all_pool_out = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # distribute data to device
            X, y, video_ids = data
            X, y = X.cuda(), y.cuda().view(-1, )

            # Procesamos el batch completo directamente
            output = model(X)

            # calcular pérdida
            loss = compute_loss(output, y)
            val_loss.append(loss.item())

            # obtener predicción (índice de la probabilidad máxima)
            y_pred = output.max(1, keepdim=True)[1]

            # recolectar resultados
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            all_video_ids.extend(video_ids)
            all_pool_out.extend(output)

    # calcular pérdida promedio
    val_loss = sum(val_loss) / len(val_loss)

    # convertir a tensores y luego a numpy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0).squeeze()
    all_pool_out = torch.stack(all_pool_out, dim=0).cpu().data.numpy()

    # identificar muestras mal clasificadas
    incorrect_indices = torch.nonzero(all_y - all_y_pred).squeeze().data
    
    # Manejo de caso si solo hay un elemento incorrecto
    if incorrect_indices.dim() == 0:
        incorrect_indices = incorrect_indices.unsqueeze(0)
        
    incorrect_video_ids = [(all_video_ids[i], int(all_y_pred[i].data)) for i in incorrect_indices]

    all_y = all_y.cpu().data.numpy()
    all_y_pred = all_y_pred.cpu().data.numpy()

    # calcular exactitud top-k
    top1acc = accuracy_score(all_y, all_y_pred)
    top3acc = compute_top_n_accuracy(all_y, all_pool_out, 3)
    top5acc = compute_top_n_accuracy(all_y, all_pool_out, 5)
    top10acc = compute_top_n_accuracy(all_y, all_pool_out, 10)
    top30acc = compute_top_n_accuracy(all_y, all_pool_out, 30)

    # --- CORRECCIÓN: Silenciamos este print también porque el main.py ya imprime el resumen ---
    # print('\nVal. set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
    #     len(all_y), val_loss, 100 * top1acc))

    return val_loss, [top1acc, top3acc, top5acc, top10acc, top30acc], all_y.tolist(), all_y_pred.tolist(), incorrect_video_ids

def compute_loss(out, gt):
    ce_loss = F.cross_entropy(out, gt)
    return ce_loss

def compute_top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    ts = truths
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / ts.shape[0]