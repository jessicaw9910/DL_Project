#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# referred to this implementation for guidance on a complete train() function
# https://inside-machinelearning.com/en/the-ideal-pytorch-function-to-train-your-model-easily/
def train(model, optimizer, scheduler, dl_train, dl_val, path, device, epochs=100):
    import time
    
    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))
    
    history = {}
    history['loss_train'] = []
    history['loss_bce'] = []
    history['loss_kld'] = []
    history['loss_val'] = [] 
    history['acc_train'] = []
    history['acc_val'] = []
    
    start_time = time.time()
    
    # save model if better than this
    best_epoch_loss_val = 100000
    
    for epoch in range(1, epochs+1):
        
        # --- Train and Evaluate Performance on Training Set ---------------------------------------------------
        
        model.train()
        
        # loss metrics
        train_loss = 0.0
        bce_loss = 0.0
        kld_loss = 0.0

        # accuracy metrics
        num_train_correct = 0
        num_train_examples = 0

        for batch in dl_train:
            
            optimizer.zero_grad()
            
            X = batch.to(device)
            X_hat = model(X)
            
            bce = F.binary_cross_entropy(X_hat, X, reduction='sum')
            kld = -0.5 * torch.mean(1 + model.z_logvar - model.z_mean.pow(2) - model.z_logvar.exp())
            loss = bce + kld

            loss.backward()
            optimizer.step()

            # loss calculations
            train_loss += loss
            bce_loss += bce
            kld_loss += kld
            
            # accuracy calcuations
            num_train_correct += sum(X_hat.argmax(axis=2) == X.argmax(axis=2)) 
            num_train_examples += X.shape[0]
            
        train_loss = train_loss / len(dl_train.dataset)
        bce_loss = bce_loss / len(dl_train.dataset)
        kld_loss = kld_loss / len(dl_train.dataset)
        train_acc = torch.mean(num_train_correct / num_train_examples)
        
        # --- Evaluate on validation set -----------------------------------------------------------------------
        
        model.eval()
        
        val_loss = 0.0
        num_val_correct = 0
        num_val_examples = 0
        
        for batch in dl_val:
            # to speed up val evaluation
            with torch.no_grad():
                X = batch.to(device)
                X_hat = model(X)

                bce = F.binary_cross_entropy(X_hat, X, reduction='sum')
                kld = -0.5 * torch.mean(1 + model.z_logvar - model.z_mean.pow(2) - model.z_logvar.exp())
                loss = bce + kld
                
                # loss calculations
                val_loss += loss
                
                # accuracy calcuations
                num_val_correct += sum(X_hat.argmax(axis=2) == X.argmax(axis=2)) 
                num_val_examples += X.shape[0]

        val_loss = val_loss / len(dl_val.dataset)
        val_acc = torch.mean(num_val_correct / num_val_examples)
        
        if epoch == 1 or epoch % 10 == 0:
            print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                  (epoch, epochs, train_loss, train_acc, val_loss, val_acc))

        # save model if validation loss best seen so far
        if val_loss < best_epoch_loss_val:
            PATH = path + str(epoch) + '_' + time.strftime("%Y%m%d-%H%M%S") + '_' + 'checkpoint.pth'
            torch.save(model.state_dict(), PATH)
            best_epoch_loss_val = val_loss
        
        history['loss_train'].append(train_loss)
        history['loss_bce'].append(bce_loss)
        history['loss_kld'].append(kld_loss)
        history['loss_val'].append(val_loss)
        history['acc_train'].append(train_acc)
        history['acc_val'].append(val_acc)

        scheduler.step(val_loss)
        
    # outside of epoch loop
    end_time = time.time()
    total_time = end_time - start_time
    time_per_epoch = total_time / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time))
    print('Time per epoch: %5.2f sec' % (time_per_epoch))
    
    return history