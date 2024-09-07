# imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import yaml
import torch

from src.model.model import WordSegmentation
from src.data_preprocessing import preprocess_data
import datetime





if __name__ == '__main__':

    # open config file
    with open('configs/vib_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data_path = config['data']['data_path']
    split_ratio = config['data']['split_ratio']
    model = config['model']['name']
    device = config['model']['device']
    parallel_devices = config['model']['parallel_devices']
    device_ids = config['model']['device_ids']
    lr = config['train']['lr']
    num_epochs = config['train']['num_epochs']
    batch_size = config['train']['batch_size']




    # config a model
    ws = WordSegmentation(data_path=data_path, model=model, device=device, 
                 lr=lr, num_epochs=num_epochs, batch_size=batch_size, parallel_devices=parallel_devices, device_ids=device_ids)

    # data preprocessing
    ws.set_datasets(preprocess_data(data_path=ws.data_path, tokenizer=ws.tokenizer, split_ratio=split_ratio))

    # train model
    ws.train_model()

    # evaluate model
    results = ws.evaluate_model()

    # save model
    now = datetime.datetime.now()
    ws.save_model(path="./models/word_segmentation"+now)