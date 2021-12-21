#!/usr/bin/env python3

import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_loss(dictionary, path):
    x = np.arange(len(dictionary['loss_train'])) + 1
    y1 = [i.cpu().detach().numpy() for i in dictionary['loss_train']]
    y2 = [i.cpu().detach().numpy() for i in dictionary['loss_val']]

    plt.scatter(x, y1, c = 'red', label='Training');
    plt.scatter(x, y1, c = 'red');
    plt.scatter(x, y2, c = 'orange', label='Validation');
    plt.plot(x, y2, c = 'orange');
    plt.xlabel('Epoch');
    plt.ylabel('Loss (BCE + KLD)');
    plt.title('Loss by Epoch');
    plt.legend();
    plt.savefig(path, bbox_inches='tight');
    plt.show();

def plot_accuracy(dictionary, path):
    x = np.arange(len(dictionary['acc_train'])) + 1
    y1 = [i.cpu().detach().numpy() for i in dictionary['acc_train']]
    y2 = [i.cpu().detach().numpy() for i in dictionary['acc_val']]

    plt.scatter(x, y1, c = 'red', label='Training');
    plt.plot(x, y1, c = 'red');
    plt.scatter(x, y2, c = 'orange', label='Validation');
    plt.plot(x, y2, c = 'orange');
    plt.xlabel('Epoch');
    plt.ylabel('Average Accuracy');
    plt.title('Average Accuracy by Epoch');
    plt.savefig(path, bbox_inches='tight');
    plt.legend();

def plot_components(dictionary, path):
    x = np.arange(len(dictionary['loss_bce'])) + 1
    y1 = [i.cpu().detach().numpy() for i in dictionary['loss_bce']]
    y2 = [i.cpu().detach().numpy() for i in dictionary['loss_kld']]

    fig, ax1 = plt.subplots();
    ax2 = ax1.twinx();
    ax1.scatter(x, y1, c='blue', label='BCE');
    ax1.plot(x, y1, c='blue');
    ax2.scatter(x, y2, c='green', label='KLD');
    ax2.plot(x, y2, c='green');
    ax1.set_xlabel("Epoch");
    ax1.set_ylabel("Binary Cross-Entropy",);
    ax2.set_ylabel("KL Divergence",);
    ax1.legend(loc='upper center');
    ax2.legend(loc='upper right');
    ax1.set_title('Components of Loss by Epoch');
    fig.savefig(path, bbox_inches='tight');
    fig.show();

def generate_statistics(list_of_smiles, training_data):
    '''Return number valid SMILES, number of unique molecules, and average number of rings'''
    from rdkit.Chem import AllChem as Chem
    
    training_data = set(training_data)
    valid_list = []
    count_molecules = 0
    cannonical_smiles = set()
    ringcnt = 0
    novelcnt = 0
    for smiles in list_of_smiles:
        if smiles == '':
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_list.append(smiles)
                count_molecules += 1
                can = Chem.MolToSmiles(mol)
                if can not in cannonical_smiles:
                    #print(can)
                    cannonical_smiles.add(can)
                    r = mol.GetRingInfo()
                    ringcnt += r.NumRings()
                    if can not in training_data:
                        novelcnt += 1
                    
        except:
            continue
    N = len(cannonical_smiles)
    ringave = 0 if N == 0 else ringcnt/N
    return count_molecules, N, novelcnt, ringave, valid_list