# coding: UTF-8
import time
import torch
import numpy as np
import pandas as pd
from train_eval import train, init_network, predict
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    # np.random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    sample_list = build_dataset(config)
    train_data = sample_list['train']['contents']
    dev_data = sample_list['dev']['contents']
    test_data = sample_list['test']['contents']
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

    # predict
    if config.predict:
        results = pd.DataFrame(columns=['text', 'label'])
        predict_labels = predict(config, model, test_iter)
        for idx in range(len(predict_labels)):
            sentences = sample_list['test']['sentences'][idx]
            label = predict_labels[idx]
            results = results.append({'text': sentences, 'label': label}, ignore_index=True)
        results.to_csv('results.txt', sep='\t', index=False, header=False)
        # print(predict_labels)
        # print(results['text'])
