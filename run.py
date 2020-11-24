# coding: UTF-8
import time
import torch
import numpy as np
import pandas as pd
from train_eval import train, init_network, predict
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
from tqdm import tqdm

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

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
    if config.predict:
      train_data, dev_data = build_dataset(config)
      train_iter = build_iterator(train_data, config)
      dev_iter = build_iterator(dev_data, config)
      test_iter = None
    else:
      train_data, dev_data, test_data = build_dataset(config)
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
        contents = []
        pad_size = config.pad_size
        with open(config.test_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                results = results.append({'text': content}, ignore_index=True)
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        cur_config = config
        cur_config.batch_size = 1
        test_iter = build_iterator(contents, cur_config)
        predict_labels = predict(config, model, test_iter)
        for idx in results.index:
            results.loc[idx, 'label'] = predict_labels[idx]
        results.to_csv('results.txt', sep='\t', index=False, header=False)
        print(predict_labels)
        print(results['text'])
