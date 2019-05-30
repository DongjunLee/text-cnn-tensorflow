# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import csv
import os
import random
import re

import numpy as np
from hbconfig import Config
import tensorflow as tf
from tqdm import tqdm


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.decode('utf-8')
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "rb").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "rb").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = ['1' for _ in positive_examples]
    negative_labels = ['0' for _ in negative_examples]
    y = positive_labels + negative_labels
    return x_text, y

def prepare_raw_data():
    print('Preparing raw data into train set and test set ...')
    raw_data_path = os.path.join(Config.data.base_path, Config.data.raw_data_path)

    data_type = Config.data.type
    if data_type == "kaggle_movie_review":
        train_path = os.path.join(raw_data_path, 'train.tsv')
        train_reader = csv.reader(open(train_path), delimiter="\t")
        prepare_dataset(dataset=list(train_reader))

    

def prepare_dataset(dataset=None, x_text=None, y=None):
    make_dir(os.path.join(Config.data.base_path, Config.data.processed_path))

    print("Path to store datasets = ",os.path.join(Config.data.base_path, Config.data.processed_path))

    filenames = ['train_X', 'train_y', 'test_X', 'test_y']
    files = []
    for filename in filenames:
        files.append(open(os.path.join(Config.data.base_path, Config.data.processed_path, filename), 'wb'))
    print("files = ",files)
    if dataset is not None:

        print("Total data length : ", len(dataset))
        test_ids = random.sample([i for i in range(len(dataset))], Config.data.testset_size)

        for i in tqdm(range(len(dataset))):
            if i == 0:
                continue

            data = dataset[i]
            X, y = data[2], data[3]


            if i in test_ids:
                files[2].write((X + "\n").encode('utf-8'))
                files[3].write((y + '\n').encode('utf-8'))
            else:
                files[0].write((X + '\n').encode('utf-8'))
                files[1].write((y + '\n').encode('utf-8'))

    else:

        print("Total data length : ", len(y))
        test_ids = random.sample([i for i in range(len(y))], Config.data.testset_size)

        for i in tqdm(range(len(y))):
            if i in test_ids:
                files[2].write((x_text[i] + "\n").encode('utf-8'))
                files[3].write((y[i] + '\n').encode('utf-8'))
            else:
                files[0].write((x_text[i] + '\n').encode('utf-8'))
                files[1].write((y[i] + '\n').encode('utf-8'))

    for file in files:
        file.close()

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words

def build_vocab(train_fname, test_fname, normalize_digits=True):
    vocab = {}
    def count_vocab(fname):
        with open(fname, 'rb') as f:
            for line in f.readlines():
                line = line.decode('utf-8')
                for token in basic_tokenizer(line):
                    if not token in vocab:
                        vocab[token] = 0
                    vocab[token] += 1

    train_path = os.path.join(Config.data.base_path, Config.data.processed_path, train_fname)
    test_path = os.path.join(Config.data.base_path, Config.data.processed_path, test_fname)

    print("train path = ",train_path)
    print("Test path = ", test_path)

    count_vocab(train_path)
    count_vocab(test_path)

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)

    dest_path = os.path.join(Config.data.base_path, Config.data.processed_path, 'vocab')
    with open(dest_path, 'wb') as f:
        f.write(('<pad>' + '\n').encode('utf-8'))
        index = 1
        for word in sorted_vocab:
            f.write((word + '\n').encode('utf-8'))
            index += 1

def load_vocab(vocab_fname):
    print("load vocab ...")
    with open(os.path.join(Config.data.base_path, Config.data.processed_path, vocab_fname), 'rb') as f:
        words = f.read().decode('utf-8').splitlines()
    return {words[i]: i for i in range(len(words))}

def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<pad>']) for token in basic_tokenizer(line)]

def token2id(data):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab'
    in_path = data
    out_path = data + '_ids'

    vocab = load_vocab(vocab_path)
    in_file = open(os.path.join(Config.data.base_path, Config.data.processed_path, in_path), 'rb')
    out_file = open(os.path.join(Config.data.base_path, Config.data.processed_path, out_path), 'wb')

    lines = in_file.read().decode('utf-8').splitlines()
    for line in lines:
        ids = []
        sentence_ids = sentence2id(vocab, line)
        ids.extend(sentence_ids)

        out_file.write(b' '.join(str(id_).encode('utf-8') for id_ in ids) + b'\n')

def process_data():
    print('Preparing data to be model-ready ...')

    build_vocab('train_X', 'test_X')

    token2id('train_X')
    token2id('test_X')

def make_train_and_test_set(shuffle=True):
    print("make Training data and Test data Start....")

    if Config.data.get('max_seq_length', None) is None:
        set_max_seq_length(['train_X_ids', 'test_X_ids'])

    train_X, train_y = load_data('train_X_ids', 'train_y')
    test_X, test_y = load_data('test_X_ids', 'test_y')

    assert len(train_X) == len(train_y)
    assert len(test_X) == len(test_y)

    print(f"train data count : {len(train_y)}")
    print(f"test data count : {len(test_y)}")

    if shuffle:
        print("shuffle dataset ...")
        train_p = np.random.permutation(len(train_y))
        test_p = np.random.permutation(len(test_y))

        return ((train_X[train_p], train_y[train_p]),
                (test_X[test_p], test_y[test_p]))
    else:
        return ((train_X, train_y),
                (test_X, test_y))

def load_data(X_fname, y_fname):
    X_input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, X_fname), 'r')
    y_input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, y_fname), 'r')

    X_data, y_data = [], []
    for X_line, y_line in zip(X_input_data.readlines(), y_input_data.readlines()):
        X_ids = [int(id_) for id_ in X_line.split()]
        y_id = int(y_line)

        if len(X_ids) == 0 or y_id >= Config.data.num_classes:
            continue

        if len(X_ids) <= Config.data.max_seq_length:
            X_data.append(_pad_input(X_ids, Config.data.max_seq_length))

        y_one_hot = np.zeros(Config.data.num_classes)
        y_one_hot[int(y_line)] = 1
        y_data.append(y_one_hot)

    print(f"load data from {X_fname}, {y_fname}...")
    return np.array(X_data, dtype=np.int32), np.array(y_data, dtype=np.int32)

def _pad_input(input_, size):
    return input_ + [0] * (size - len(input_))

def set_max_seq_length(dataset_fnames):

    max_seq_length = Config.data.get('max_seq_length', 10)

    for fname in dataset_fnames:
        input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, fname), 'r')

        for line in input_data.readlines():
            ids = [int(id_) for id_ in line.split()]
            seq_length = len(ids)

            if seq_length > max_seq_length:
                max_seq_length = seq_length

    Config.data.max_seq_length = max_seq_length
    print(f"Setting max_seq_length to Config : {max_seq_length}")

def make_batch(data, buffer_size=10000, batch_size=64, scope="train"):

    class IteratorInitializerHook(tf.train.SessionRunHook):
        """Hook to initialise data iterator after Session is created."""

        def __init__(self):
            super(IteratorInitializerHook, self).__init__()
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            """Initialise the iterator after the session has been created."""
            self.iterator_initializer_func(session)


    def get_inputs():

        iterator_initializer_hook = IteratorInitializerHook()

        def train_inputs():
            with tf.name_scope(scope):

                X, y = data

                # Define placeholders
                input_placeholder = tf.placeholder(
                    tf.int32, [None, Config.data.max_seq_length])
                output_placeholder = tf.placeholder(
                    tf.int32, [None, Config.data.num_classes])

                # Build dataset iterator
                dataset = tf.data.Dataset.from_tensor_slices(
                    (input_placeholder, output_placeholder))

                if scope == "train":
                    dataset = dataset.repeat(None)  # Infinite iterations
                else:
                    dataset = dataset.repeat(1)  # 1 Epoch
                # dataset = dataset.shuffle(buffer_size=buffer_size)
                dataset = dataset.batch(batch_size)

                iterator = dataset.make_initializable_iterator()
                next_X, next_y = iterator.get_next()

                tf.identity(next_X[0], 'input_0')
                tf.identity(next_y[0], 'target_0')

                # Set runhook to initialize iterator
                iterator_initializer_hook.iterator_initializer_func = \
                    lambda sess: sess.run(
                        iterator.initializer,
                        feed_dict={input_placeholder: X,
                                   output_placeholder: y})

                # Return batched (features, labels)
                return next_X, next_y

        # Return function and hook
        return train_inputs, iterator_initializer_hook

    return get_inputs()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    prepare_raw_data()
    process_data()
