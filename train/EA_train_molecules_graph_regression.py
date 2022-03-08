"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import dgl
import networkx as nx

import torch
import torch.nn as nn
import math

from train.metrics import MAE

def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    
    # modif
    
    DISPLAY = False
    scores = np.zeros((1,1))
    
    if epoch%10==0:
        
        DISPLAY = True
        
        mapping = {'C': 0,'O': 1,'N': 2,'F': 3,'C H1': 4,'S': 5,'Cl': 6,'O -': 7,'N H1 +': 8,'Br': 9,'N H3 +': 10,'N H2 +': 11,'N +': 12,'N -': 13,'S -': 14,'I': 15,'P': 16,'O H1 +': 17,'N H1 -': 18,'O +': 19,'S +': 20,'P H1': 21,'P H2': 22,'C H2 -': 23,'P +': 24,'S H1 +': 25,'C H1 -': 26,'P H1 +': 27}
        mapping = {v: k for k, v in mapping.items()}
        iter_to_show = np.random.randint(low=0, high=78)
        item_in_batch_to_show = np.random.randint(low=0, high=128)
    
    # end modif
    
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None
            
        try:
            batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
        except:
            batch_wl_pos_enc = None

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
        
        # modif
        
        if DISPLAY==True and iter==iter_to_show:
            graph = dgl.unbatch(batch_graphs)[item_in_batch_to_show]
            x = graph.ndata['feat'].to(device)
            z = graph.ndata['z'].to(device)
            z_mean = torch.squeeze(torch.mean(z, dim=1))
            scores = z_mean.cpu().detach().numpy()

            G = dgl.to_networkx(graph.cpu())

            plt.figure(1, figsize=(15,10))
            print('\n @ Epoch {} ------- \n\n Showing \n - {}th element out of 128 \n - in batch {} out of 78\n'.format(epoch, item_in_batch_to_show, iter_to_show))
            nx.draw(G,
                    with_labels = True,
                    node_color = scores,
                    node_size = 600,
                    labels = {i : mapping[x[i].item()] for i in range(x.shape[0])},
                    cmap = plt.cm.coolwarm,
                    pos = nx.spring_layout(G, iterations=100))
            plt.show()
            print('\n Values of scores : \n {} \n\n _________ \n\n'.format(scores))
            print('Expected solubility : {} .\nSolubility predicted : {} . \n\n --------------------------------------------------\n\n'.format(batch_targets[0].item(), batch_scores[0].item()))
            
            # end modif
            
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer, scores

def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None
            
            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None
                
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae

