#!/usr/bin/env python3

import numpy as np
import torch
import pickle as pkl

from utils import *
from model import ConvEncoder, GRUDecoder, ChemVAE
from trainer import train

def main(args):
    df = import_data(args.input, args.path)
    # previously pre-processed version on path ('data/zinc.csv') with below command
    # df['SELFIES'] = df['SMILES'].apply(convert_smiles2selfies)

    # GitHub data directory contains only tar.gz version
    # import pandas as pd
    # df = pd.read_csv('../data/zinc.tar.gz', compression='gzip', header=0, sep=',', error_bad_lines=False)
    # df.columns[0] = 'SMILES'

    X_train, X_test = return_splits(df, args.train, args.val, args.cols, args.colc)
    char2idx, idx2char, train_idx, test_idx = create_data(X_train, X_test, colname=args.colc)
    train_oh, test_oh = check_conversions(idx2char, train_idx, X_train, test_idx, X_test)

    path = args.model + args.colc + '/'

    n_length = train_oh.shape[1]
    n_char = train_oh.shape[2]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    enc = ConvEncoder(args.latent, n_length, n_char).to(device)
    dec = GRUDecoder(args.latent, n_length, n_char).to(device)
    model = ChemVAE(enc, dec).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    if args.dynlr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.8, patience = 3, min_lr = 0.0001)

    X_train = torch.from_numpy(train_oh.astype(np.float32))
    X_test = torch.from_numpy(test_oh.astype(np.float32))

    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=args.batch, shuffle=True, num_workers=6, drop_last = True)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=args.batch, shuffle=True, num_workers=6, drop_last = True)

    history = train(model, optimizer, train_loader, test_loader, path, device, epochs=args.epochs, sched=scheduler)

    if args.output:
        # save char2idx dictionary as pkl
        filename = args.colc + '_char2idx_dict'
        outfile = open(filename, 'wb')
        pkl.dump(char2idx, outfile)
        outfile.close()

        # save history dictionary as pkl
        filename = args.colc + '_history_dict'
        outfile = open(filename, 'wb')
        pkl.dump(history, outfile)
        outfile.close()

def parsearg_utils():
    import argparse

    parser = argparse.ArgumentParser(description='Download and pre-process SMILE strings.')

    parser.add_argument('-i','--input', help='URL or path where csv of SMILE string data can be found (str)', default='https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv', type=str)
    parser.add_argument('-t','--train', help='Number of training samples (int)', default=1000, type=int)
    parser.add_argument('-v','--val', help='Number of testing/validation samples (int)', default=100, type=int)
    parser.add_argument('-p','--path', help='Path to save data as file including name.csv (str)', default=None, type=str)
    parser.add_argument('-o','--output', help='If True returns processed data (bool)', default=True, type=bool)
    parser.add_argument('-s','--cols', help='Name of column with split allocations (str)', default='SPLIT', type=str)
    ## set to 'SELFIES' to train on SELFIES strings instead
    parser.add_argument('-c','--colc', help='Name of column with chemical entitites (str)', default='SMILES', type=str)
    parser.add_argument('-l','--lr', help='Learning rate (float)', default=0.001, type=float)
    parser.add_argument('-d','--dynlr', help='Indicates if dynamic learning rate scheduler in use (bool)', default=True, type=bool)
    parser.add_argument('-b','--batch', help='Batch size (int)', default=200, type=int)
    parser.add_argument('-z','--latent', help='Latent dimension (int)', default=488, type=int)
    parser.add_argument('-n','--seed', help='Seed (int)', default=123, type=int)
    parser.add_argument('-m','--model', help='Path to store model (str)', default='../weights/', type=str)
    parser.add_argument('-e','--epochs', help='Epochs (int)', default=100, type=int)    

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    arguments = parsearg_utils()
    main(arguments)