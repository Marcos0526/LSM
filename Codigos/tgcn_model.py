#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np


class GraphConvolution_att(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init_A=0, num_nodes=None):
        super(GraphConvolution_att, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # AHW
        support = torch.matmul(input, self.weight)  # HW
        output = torch.matmul(self.att, support)  # g
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):

    def __init__(self, in_features, p_dropout, bias=True, is_resi=True, num_nodes=None):
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.is_resi = is_resi

        # CORRECCIÓN AQUÍ: Pasar num_nodes
        self.gc1 = GraphConvolution_att(in_features, in_features, num_nodes=num_nodes)
        self.bn1 = nn.BatchNorm1d(num_nodes * in_features)

        # CORRECCIÓN AQUÍ: Pasar num_nodes
        self.gc2 = GraphConvolution_att(in_features, in_features, num_nodes=num_nodes)
        self.bn2 = nn.BatchNorm1d(num_nodes * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        if self.is_resi:
            return y + x
        else:
            return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_muti_att(nn.Module):
    # NUEVO: Añadido el parámetro `classes=None` al constructor
    def __init__(self, input_feature, hidden_feature, num_class, p_dropout, num_stage=1, is_resi=True, num_nodes=None, classes=None):
        super(GCN_muti_att, self).__init__()
        self.num_stage = num_stage
        
        # NUEVO: Guardar las clases. Si no se especifican, se crean índices numéricos por defecto.
        if classes is not None:
            assert len(classes) == num_class, f"Error: Esperadas {num_class} clases, pero se recibieron {len(classes)}."
            self.classes = classes
        else:
            self.classes = [str(i) for i in range(num_class)]

        # CORRECCIÓN AQUÍ: Pasar num_nodes
        self.gc1 = GraphConvolution_att(input_feature, hidden_feature, num_nodes=num_nodes)
        self.bn1 = nn.BatchNorm1d(num_nodes * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            # CORRECCIÓN AQUÍ: Pasar num_nodes al crear el bloque
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, is_resi=is_resi, num_nodes=num_nodes))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        self.fc_out = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        out = torch.mean(y, dim=1)
        
        # ---> AÑADIR DROPOUT AQUÍ <---
        out = self.do(out) 
        
        out = self.fc_out(out)

        return out

    # NUEVO: Método para traducir la salida numérica a nombres de clases legibles
    def get_predicted_classes(self, logits):
        """Devuelve los nombres de las clases predecidas basados en la salida del modelo (logits)"""
        _, predicted_indices = torch.max(logits, 1)
        return [self.classes[idx] for idx in predicted_indices]


if __name__ == '__main__':
    num_samples = 32
    n_nodes = 75 # Pon aquí el número de puntos que usarás realmente (ej. 75)
    
    # NUEVO: Definir nuestras clases en formato lista
    nombres_de_clases = ["a", "b", "c"]
    num_classes = len(nombres_de_clases) # En este ejemplo son 3

    # NUEVO: Le pasamos 'classes=nombres_de_clases' y ajustamos 'num_class'
    model = GCN_muti_att(input_feature=num_samples*2, hidden_feature=256,
                         num_class=num_classes, p_dropout=0.3, num_stage=2, 
                         num_nodes=n_nodes, classes=nombres_de_clases)
    
    x = torch.ones([2, n_nodes, num_samples*2])
    salida = model(x)
    
    print("Formato de salida:", salida.size())
    
    # Imprimir las clases guardadas en estilo ["a", "b", "c"]
    print("Clases guardadas en el modelo:", model.classes)
    
    # Probar la predicción usando la nueva función
    predicciones = model.get_predicted_classes(salida)
    print("Predicción para cada muestra del batch:", predicciones)