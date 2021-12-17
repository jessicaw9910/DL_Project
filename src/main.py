#!/usr/bin/env python3

# import numpy as np
from utils import *

def main(args):
    df = import_data(args.input, args.path)
    # previously pre-processed version on path ('data/zinc.csv')
    # df['SELFIES'] = df['SMILES'].apply(convert_smiles2selfies)
    X_train, X_test = return_splits(df, args.train, args.val, args.cols, args.colc)

    if args.colc == 'SMILES':
        char2idx, idx2char = create_vocab(X_train)
        train_idx = [convert_str2num(i, char2idx) for i in X_train]
        test_idx = [convert_str2num(i, char2idx) for i in X_test]
    else:
        # SELFIE string needs to be pre-proccessed into tokens
        train_token = tokenize_selfie(X_train)
        test_token = tokenize_selfie(X_test)
        char2idx, idx2char = create_vocab(train_token)
        train_idx = [convert_str2num(i, char2idx) for i in train_token]
        test_idx = [convert_str2num(i, char2idx) for i in test_token]

    X_train = [tokenize_selfie(i) for i in X_train if args.colc == 'SELFIES']
    X_test = [tokenize_selfie(i) for i in X_test if args.colc == 'SELFIES']
    char2idx, idx2char = create_vocab(X_train)
    train_idx = [convert_str2num(i, char2idx) for i in X_train]
    test_idx = [convert_str2num(i, char2idx) for i in X_test]

    train_char = [convert_num2str(i, idx2char) for i in train_idx]
    print("There are %d training index conversion errors" % (sum([train_char[i] != char for i, char in enumerate(X_train)])))

    test_char = [convert_num2str(i, idx2char) for i in test_idx]
    print("There are %d testing index conversion errors" % (sum([test_char[i] != char for i, char in enumerate(X_test)])))

    print("")

    train_oh = convert_num2onehot(train_idx)
    test_oh = convert_num2onehot(test_idx)

    train_oh_idx = convert_onehot2num(train_oh)
    print("There are %d training one-hot conversion errors" % (sum([train_oh_idx[i] != char for i, char in enumerate(train_idx)])))

    test_oh_idx = convert_onehot2num(test_oh)
    print("There are %d testing one-hot conversion errors" % (sum([test_oh_idx[i] != char for i, char in enumerate(test_idx)])))

    if args.output:
        return X_train, X_test, char2idx, idx2char, train_idx, test_idx, train_oh, test_oh

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Run Chemical VAE.')

    parser.add_argument('-i','--input', help='URL or path where csv of SMILE string data can be found (str)', default='https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv', type=str)
    parser.add_argument('-t','--train', help='Number of training samples (int)', default=1000, type=int)
    parser.add_argument('-v','--val', help='Number of testing/validation samples (int)', default=100, type=int)
    parser.add_argument('-p','--path', help='Path to save data as file including name.csv (str)', default=None, type=str)
    parser.add_argument('-o','--output', help='If True returns processed data (bool)', default=False, type=bool)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    arguments = parse_arguments()
    main_utils(arguments)