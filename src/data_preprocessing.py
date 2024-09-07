import pandas as pd
import numpy as np
import time
import random
from src.utilities import tokenize_and_label_function
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import os



# def preprocess_data(data_path, tokenizer, model='byt5', split_ratio=0.9):
def preprocess_data(data_path, tokenizer, split_ratio=0.9):
    """
    Tokenizes the input and target texts from the dataset.

    Parameters:
    - data_path (str): The path to the CSV file containing the dataset.
    - tokenizer (AutoTokenizer): The tokenizer to use for tokenization.
    - model (str): The model to use for tokenization.
    - split_ratio (float): The ratio of training data to evaluation data. Default is 0.9.
    
    This method reads the dataset in chunks (if data_path is provided) or directly from the DataFrame (if df is provided).
    It tokenizes the text data and splits it into training and evaluation datasets.
    """
    
    start_time = time.time()
    print(f'start preporocess data')
    # if there is already a tokenized dataset, load it and return it. note that the tokenized dataset is saved as an arrow file in a folder f'{data_path[:-4]}_tokenized.csv' with the name data-0000k-of-0000n.arrow where there are ks from 1 to n
    if os.path.exists(f'{data_path[:-4]}_tokenized'):
    # if False:
        print(f'loading tokenized dataset')
        # tokenized_dataset = load_dataset('arrow', data_files=f'{data_path[:-4]}_tokenized.csv/data-00000-of-00001.arrow', split='train').remove_columns('token_type_ids')
        tokenized_dataset = load_dataset('arrow', data_files=f'{data_path[:-4]}_tokenized/data-00000-of-00001.arrow', split='train')
        # if there is a column named 'token_type_ids' in the dataset, remove it
        if 'token_type_ids' in tokenized_dataset.column_names:
            tokenized_dataset = tokenized_dataset.remove_columns('token_type_ids')
        print(f'loaded tokenized dataset after {time.time()-start_time}')



    else:
        dataset = load_dataset('csv', data_files=data_path, split='train')
        print(f'loaded dataset after {time.time()-start_time}')

        tokenized_dataset = dataset.map(tokenize_and_label_function, batched=False,load_from_cache_file=False, fn_kwargs={'tokenizer': tokenizer})

        # save the tokenized dataset to a csv file in the same directory as the original dataset
        tokenized_dataset.save_to_disk(f'{data_path[:-4]}_tokenized.csv')
    print(f'tokenized dataset after {time.time()-start_time}')


    # remove the columns'input_text', 'target_text' from the dataset
    no_str_tokenized_dataset = tokenized_dataset.remove_columns(['input_text', 'target_text'])


    split_dataset = no_str_tokenized_dataset.train_test_split(test_size=1-split_ratio)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print(f'train_dataset: {train_dataset}')

    print(f'finished preprocess data after {time.time()-start_time}')

    return train_dataset, eval_dataset, tokenized_dataset


