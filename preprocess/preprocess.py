import sys
import glob 
import os 
import json 
import kaldi_io
import pickle 


def load_data(dir):
    feature = {}
    for ark_file in sorted(glob.glob(os.path.join(dir, '*.ark'))):
        print(f'loading {ark_file}...')
        for key, mat in kaldi_io.read_mat_ark(os.path.join(dir, ark_file)):
            feature[key] = mat

    with open(os.path.join(dir, 'data_unigram5000.json')) as f:
        data = json.load(f)
    return feature, data

def load_dict(dict_path):
    vocab_dict = {'<BLANK>': 0}
    with open(dict_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
        sym, ind = line.strip().split(maxsplit=1)
        vocab_dict[sym] = len(vocab_dict)
    return vocab_dict

def get_token_ids(data_dict, vocab_dict):
    data = {}
    for utt_id in data_dict['utts']:
        tokens = data_dict['utts'][utt_id]['output'][0]['token'].split()
        token_ids = [vocab_dict[token] for token in tokens if token in vocab_dict]
        data[utt_id] = token_ids
    return data

def merge_data(feature, token_ids):
    data = {}
    for utt_id in feature:
        data[utt_id] = {'feature': feature[utt_id], 'token_ids': token_ids[utt_id]}
    return data

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python3 preprocess.py [root_dir] [dict_path] [output_dir]')

    root_dir = sys.argv[1]
    dict_path = sys.argv[2]
    output_dir = sys.argv[3]
    # deltatrue/deltafalse
    in_dir = 'deltafalse'

    train = 'train_100'
    dev = 'dev_clean'
    test = 'test_clean'

    vocab_dict = load_dict(dict_path)
    dict_output_path = os.path.join(output_dir, 'vocab_dict.pkl')

    with open(dict_output_path, 'wb') as f:
        pickle.dump(vocab_dict, f)

    dsets = [train, dev, test]
    for i, dset in enumerate(dsets):
        print(f'processing {dset}...')
        dir = os.path.join(root_dir, f'{dset}/{in_dir}')
        print('load data...')
        feature, data_dict = load_data(dir)
        token_ids = get_token_ids(data_dict, vocab_dict)
        data = merge_data(feature, token_ids)
        print(f'total utterances={len(data)}')
        print('dump data...')
        data_output_path = os.path.join(output_dir, f'{dset}.pkl')
        with open(data_output_path, 'wb') as f:
            pickle.dump(data, f)

