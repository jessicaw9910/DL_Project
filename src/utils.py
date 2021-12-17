#!/usr/bin/env python3

import numpy as np

def import_data(url, csv_path=None):
    '''
    Import download and process ZINC data set into training and testing samples

    Args:
        url (string): web address for csv data
    
    Returns:
        df (pd.dataframe): returns dataframe that has been shuffled
    '''    
    import pandas as pd

    df = pd.read_csv(url)

    df.sample(frac=1, random_state=123)

    if csv_path is not None:
        df.to_csv(csv_path, index=False)

    return df

def return_splits(df, n_train=1000, n_test=100, col_split='SPLIT', col_chem='SMILES'):
    '''
    Create training and testing samples if default data set used

    Args:
        df (pd.dataframe): dataframe that has been shuffled, needs a column where train/test labeled and column for molecules
    
    Returns:
        train (list): list of SMILE strings for training
        test (list): list of SMILE strings for testing
    '''    
    import pandas as pd

    df = df.dropna()

    df_train = df[df[col_split] == "train"].iloc[0:n_train, :]
    df_test = df[df[col_split] == "test"].iloc[0:n_test, :]

    train = df_train[col_chem].tolist()
    test = df_test[col_chem].tolist()

    return train, test

def convert_smiles2selfies(SMILE):
    '''
    Convert SMILE strings to SELFIE strings
    
    Args:
        SMILE (string): SMILE string for a single molecule
    
    Returns:
        SELFIE (string or float): SELFIE string for a single molecules, float if nan   
    '''
    import selfies as sf
    
    try:
        SELFIE = sf.encoder(SMILE)
    except sf.EncoderError:
        SELFIE = float("NaN")

    return SELFIE

def tokenize_selfie(list_selfie):
    '''
    Convert a list of SELFIE strings to a list of list of tokenized strings
    Args:
        list_selfie (list): list of SELFIE strings 
    
    Returns:
        output (list): list of list of tokenized strings
    '''
    import selfies as sf

    output = [list(sf.split_selfies(i)) for i in list_selfie]

    return output

def create_vocab(list_chem, embedding='SMILES'):
    '''
    Convert list of strings to vocabulary
    
    Args:
        list_chem (list): list of strings if SMILE and list of tokenized list of list strings if SELFIE 
        embedding (str): specifies whether embedding is SMILE or SELFIE string
    
    Returns:
        char2idx (dict): character (str) as key and index (int) as pair
        idx2char (dict): index (int) as key and character (str) as pair
    '''
    char_set = set()

    if embedding == 'SMILES':
        for string in list_chem:
            char_set.update(string)
    else:
        for string in list_chem:
            for char in string:
                char_set.update(char)

    # adding markers of start, end, unknown, and padding
    char_list = list(sorted(char_set)) + ['<beg>', '<end>', '<unk>', '<pad>']
    char2idx = {char:idx for idx, char in enumerate(char_list)}
    idx2char = {y:x for x,y in char2idx.items()}
    
    return char2idx, idx2char

def convert_char2idx(char, dictionary):
    '''
    Convert single character key to single index value
    
    Args:
        char (str): single character key input
        dictionary (dict): character (key) to index (value) pair dictionary
    
    Returns:
        idx (int): single index value output
    '''
    if char not in dictionary:
        # use <unk> index if char not recognized
        idx = int(dictionary['<unk>'])
        return idx
    else:
        idx = int(dictionary[char])
        
    return idx

def convert_idx2char(idx, dictionary):
    '''
    Convert single index key to single character value
    
    Args:
        idx (int): single index key input
        dictionary (dict): index (key) to character (value) pair dictionary
    
    Returns:
        char (str): single character value output
    '''
    # use <unk> if not recognized
    if idx not in dictionary:
        # invert key-value pairs and idx of <unk>
        char = str(dictionary[{y:x for x,y in dictionary.items()}['<unk>']])
        return char
    else:
        char = str(dictionary[idx])
        
    return char

def convert_str2num(string, dictionary, start=True, end=True, padding=True, length=60):
    '''
    Converts a string to a list of index numbers
    
    Args:
        string (str/list): SMILE string or tokenized list of SELFIE strings corresponding to chemical of interest
        start (bool): indicates if start character index should be added
        end (bool): indicates if start character index should be added
        padding (bool): indicates if padding in use
        length (int): size of output with padding, if in use
        dictionary (dict): dictionary with character keys and index values (char2idx)
    
    Returns:
        output (list): list of indices converted from characters in string
    '''
    from itertools import repeat

    output = [convert_char2idx(char, dictionary) for char in string]

    # add start
    if start:
        output.insert(0, dictionary['<beg>'])
    
    # add end
    if end:
        output.append(dictionary['<end>'])

    # add padding 
    n = len(output)
    if padding and (n < length):
        output.extend(repeat(dictionary['<pad>'], length - n))
    if padding and (n > length):
        print("Warning: output exceeds target length of %i with a length of %i" % (length, n))
        
    return output

def convert_num2str(number, dictionary, start=True, end=True, padding=True):
    '''
    Converts a string to a list of index numbers
    
    Args:
        number (list): list of ints corresponding to index (key)
        start (bool): indicates if start character index should be remove
        end (bool): indicates if start character index should be removed
        padding (bool): indicates if padded values should be removed
        dictionary (dict): dictionary with index keys and character values (idx2char)
    
    Returns:
        output (string): string of characters converted from indices in number
    '''
    if len(number) == 0:
        return ''

    output = [convert_idx2char(idx, dictionary) for idx in number]
    
    if start:
        try:
            output = list(filter(('<beg>').__ne__, output))
        except:
            pass
    if end:
        try:
            output = list(filter(('<end>').__ne__, output))
        except:
            pass
    if padding:
        try:
            output = list(filter(('<pad>').__ne__, output))
        except:
            pass
    
    output = ''.join(output)
        
    return output

def convert_num2onehot(number):
    '''
    Converts a list of index numbers to a one-hot encoded matrix
    
    Args:
        number (list): list of ints corresponding to index
    
    Returns:
        output (np.array): one-hot encoded matrix of dimensions (n_samples x n_length x n_char)
    '''
    num_array = np.array(number)

    n_samples = num_array.shape[0] ## number of samples
    n_length = num_array.shape[1] ## max sequence length
    n_char = num_array.max() + 1 ## number of unique characters

    output = np.zeros((n_samples, n_length, n_char))

    for n in range(n_samples):
        num_list = num_array[n, :]
        for i, j in enumerate(num_list):
            output[n, i, j] = 1

    # test that all values = 50
    # output.sum(axis=1).sum(axis=1) 
    
    return output

def convert_onehot2num(matrix):
    '''
    Converts a one-hot encoded matrix to a list of index numbers
    
    Args:
        matrix (np.array): one-hot encoded matrix of dimensions (n_samples x n_length x n_char)
    
    Returns:
        output (list): list of ints corresponding to index
    '''
    n_samples = matrix.shape[0] ## number of samples
    n_length = matrix.shape[1] ## max sequence length
    
    output = []
    
    for n in range(n_samples):
        temp = []
        for m in range(n_length):
            temp.append(int(np.where(matrix[n, m, :] == 1)[0]))
        output.append(temp)
        
    return output

def check_conversions(dictionary, train_num, train, test_num, test):
    '''
    Ensure forward and backward conversion from string > index number > one-hot encoded
    
    Args:
        dictionary (dict): idx2char dictionary
        train_num (list): list of idx for training set
        train (list): list of strings of training set
        test_num (list): list of idx for training set
        test (list): list of strings of test set
    
    Returns:
        train_oh (np.ndarray): returns one-hot array (n_samples x n_length x n_char) for training data
        test_oh (np.ndarray): returns one-hot array (n_samples x n_length x n_char) for training data
    '''
    train_char = [convert_num2str(i, dictionary) for i in train_num]
    print("There are %d training index conversion errors" % (sum([train_char[i] != char for i, char in enumerate(train)])))

    test_char = [convert_num2str(i, dictionary) for i in test_num]
    print("There are %d testing index conversion errors" % (sum([test_char[i] != char for i, char in enumerate(test)])))

    print("")

    train_oh = convert_num2onehot(train_num)
    test_oh = convert_num2onehot(test_num)

    train_oh_idx = convert_onehot2num(train_oh)
    print("There are %d training one-hot conversion errors" % (sum([train_oh_idx[i] != char for i, char in enumerate(train_num)])))

    test_oh_idx = convert_onehot2num(test_oh)
    print("There are %d testing one-hot conversion errors" % (sum([test_oh_idx[i] != char for i, char in enumerate(test_num)])))
    
    return train_oh, test_oh

def create_data(X_train, X_test, colname='SMILES'):
    if colname == 'SMILES':
        char2idx, idx2char = create_vocab(X_train)
        train_idx = [convert_str2num(i, char2idx) for i in X_train]
        test_idx = [convert_str2num(i, char2idx) for i in X_test]

    else:
        train_token = tokenize_selfie(X_train)
        test_token = tokenize_selfie(X_test)
        char2idx, idx2char = create_vocab(train_token)
        train_idx = [convert_str2num(i, char2idx) for i in train_token]
        test_idx = [convert_str2num(i, char2idx) for i in test_token]

    return char2idx, idx2char, train_idx, test_idx

def main_utils(args):
    df = import_data(args.input, args.path)
    # previously pre-processed version on path ('data/zinc.csv')
    # df['SELFIES'] = df['SMILES'].apply(convert_smiles2selfies)

    X_train, X_test = return_splits(df, args.train, args.val, args.cols, args.colc)
    char2idx, idx2char, train_idx, test_idx = create_data(X_train, X_test, colname=args.colc)
    train_oh, test_oh = check_conversions(idx2char, train_idx, X_train, test_idx, X_test)

    if args.output:
        return X_train, X_test, char2idx, idx2char, train_idx, test_idx, train_oh, test_oh

def parsearg_utils():
    import argparse

    parser = argparse.ArgumentParser(description='Download and pre-process SMILE strings.')

    parser.add_argument('-i','--input', help='URL or path where csv of SMILE string data can be found (str)', default='https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv', type=str)
    parser.add_argument('-t','--train', help='Number of training samples (int)', default=1000, type=int)
    parser.add_argument('-v','--val', help='Number of testing/validation samples (int)', default=100, type=int)
    parser.add_argument('-p','--path', help='Path to save data as file including name.csv (str)', default=None, type=str)
    parser.add_argument('-o','--output', help='If True returns processed data (bool)', default=False, type=bool)
    parser.add_argument('-s','--cols', help='Name of column with split allocations (str)', default='SPLIT', type=str)
    ## set to 'SELFIES' to train on SELFIES strings instead
    parser.add_argument('-c','--colc', help='Name of column with chemical entitites (str)', default='SMILES', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    arguments = parsearg_utils()
    main_utils(arguments)