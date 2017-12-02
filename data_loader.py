# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import random
import re

import numpy as np
from hbconfig import Config


def get_lines():
    id2line = {}
    file_path = os.path.join(Config.data.base_path, Config.data.line_fname)
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.decode('iso-8859-1').split(' +++$+++ ')
            if len(parts) == 5:
                if parts[4][-1] == '\n':
                    parts[4] = parts[4][:-1]
                id2line[parts[0]] = parts[4]
    return id2line


def get_convos():
    """ Get conversations from the raw data """
    file_path = os.path.join(Config.data.base_path, Config.data.conversation_fname)
    convos = []
    with open(file_path, 'rb') as f:
        for line in f.readlines():
            parts = line.decode('iso-8859-1').split(' +++$+++ ')
            if len(parts) == 4:
                convo = []
                for line in parts[3][1:-2].split(', '):
                    convo.append(line[1:-1])
                convos.append(convo)

    return convos


def cornell_question_answers(id2line, convos):
    """ Divide the dataset into two sets: questions and answers. """
    questions, answers = [], []
    for convo in convos:
        for index, line in enumerate(convo[:-1]):
            questions.append(id2line[convo[index]])
            answers.append(id2line[convo[index + 1]])
    assert len(questions) == len(answers)
    return questions, answers


def twitter_question_answers():
    """ Divide the dataset into two sets: questions and answers. """
    file_path = os.path.join(Config.data.base_path, Config.data.line_fname)

    twitter_corpus = []
    with open(file_path, 'rb') as f:
        for line in f.readlines():
            line = line.decode('utf-8')

            if line[-1] == '\n':
                twitter_corpus.append(line[:-1])
            else:
                twitter_corpus.append(line)

    questions = twitter_corpus[::2] # even is question
    answers = twitter_corpus[1::2] # odd is answer

    assert len(questions) == len(answers)
    return questions, answers


def prepare_dataset(questions, answers):
    # create path to store all the train & test encoder & decoder
    make_dir(Config.data.processed_path)

    # random convos to create the test set
    test_ids = random.sample([i for i in range(len(questions))], Config.data.testset_size)

    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    files = []
    for filename in filenames:
        files.append(open(os.path.join(Config.data.processed_path, filename), 'wb'))

    for i in range(len(questions)):
        if i in test_ids:
            files[2].write((questions[i] + "\n").encode('utf-8'))
            files[3].write((answers[i] + '\n').encode('utf-8'))
        else:
            files[0].write((questions[i] + '\n').encode('utf-8'))
            files[1].write((answers[i] + '\n').encode('utf-8'))

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


def build_vocab(in_fname, out_fname, normalize_digits=True):
    vocab = {}
    def count_vocab(fname):
        with open(fname, 'rb') as f:
            for line in f.readlines():
                line = line.decode('utf-8')
                for token in basic_tokenizer(line):
                    if not token in vocab:
                        vocab[token] = 0
                    vocab[token] += 1

    in_path = os.path.join(Config.data.processed_path, in_fname)
    out_path = os.path.join(Config.data.processed_path, out_fname)

    count_vocab(in_path)
    count_vocab(out_path)

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)

    dest_path = os.path.join(Config.data.processed_path, 'vocab')
    with open(dest_path, 'wb') as f:
        f.write(('<pad>' + '\n').encode('utf-8'))
        f.write(('<unk>' + '\n').encode('utf-8'))
        f.write(('<s>' + '\n').encode('utf-8'))
        f.write(('<\s>' + '\n').encode('utf-8'))
        index = 4
        for word in sorted_vocab:
            if vocab[word] < Config.data.word_threshold:
                pass
            f.write((word + '\n').encode('utf-8'))
            index += 1


def load_vocab(vocab_fname):
    print("load vocab ...")
    with open(os.path.join(Config.data.base_path, Config.data.processed_path, vocab_fname), 'rb') as f:
        words = f.read().decode('utf-8').splitlines()
    return {words[i]: i for i in range(len(words))}


def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)]


def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab'
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    vocab = load_vocab(vocab_path)
    in_file = open(os.path.join(Config.data.processed_path, in_path), 'rb')
    out_file = open(os.path.join(Config.data.processed_path, out_path), 'wb')

    lines = in_file.read().decode('utf-8').splitlines()
    for line in lines:
        if mode == 'dec':  # we only care about '<s>' and </s> in decoder
            ids = [vocab['<s>']]
        else:
            ids = []

        sentence_ids = sentence2id(vocab, line)
        ids.extend(sentence_ids)
        if mode == 'dec':
            ids.append(vocab['<\s>'])

        out_file.write(b' '.join(str(id_).encode('cp1252') for id_ in ids) + b'\n')


def prepare_raw_data():
    print('Preparing raw data into train set and test set ...')

    data_type = Config.data.get('type', 'cornell-movie')
    if data_type == "cornell-movie":
        id2line = get_lines()
        convos = get_convos()
        questions, answers = cornell_question_answers(id2line, convos)
    elif data_type == "twitter":
        questions, answers = twitter_question_answers()
    elif data_type == "all":
        pass
    else:
        raise ValueError(f"Unknown data_type, {data_type}")

    prepare_dataset(questions, answers)

def process_data():
    print('Preparing data to be model-ready ...')

    build_vocab('train.enc', 'train.dec')

    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')


def make_train_and_test_set():
    print("make Training data and Test data Start....")

    train_X, train_y = load_data('train_X', 'train_y')
    test_X, test_y = load_data('test_X', 'test_y')

    if len(train_X) == len(train_y) and len(test_X) == len(test_y):
        print(f"train data count : {len(train_X)}")
        print(f"test data count : {len(test_X)}")
        return train_X, test_X, train_y, test_y
    else:
        train_count = min(len(train_X), len(train_y))
        test_count = min(len(test_X), len(test_y))

        print(f"train data count : {train_count}")
        print(f"test data count : {test_count}")

        return train_X[:train_count], test_X[:test_count], train_y[:train_count], test_y[:test_count]


def make_train_and_test_set2():
    print("make Training data and Test data Start....")

    train_X, train_y = load_data('train_ids.enc', 'train_ids.dec')
    test_X, test_y = load_data('test_ids.enc', 'test_ids.dec')

    if len(train_X) == len(train_y) and len(test_X) == len(test_y):
        print(f"train data count : {len(train_X)}")
        print(f"test data count : {len(test_X)}")
        return train_X, test_X, train_y, test_y
    else:
        train_count = min(len(train_X), len(train_y))
        test_count = min(len(test_X), len(test_y))

        print(f"train data count : {train_count}")
        print(f"test data count : {test_count}")

        return train_X[:train_count], test_X[:test_count], train_y[:train_count], test_y[:test_count]

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
        input_data = open(os.path.join(Config.data.processed_path, fname), 'r')

        for line in input_data.readlines():
            ids = [int(id_) for id_ in line.split()]
            seq_length = len(ids)

            if seq_length > max_seq_length:
                max_seq_length = seq_length

    Config.data.max_seq_length = max_seq_length
    print(f"Setting max_seq_length to Config : {max_seq_length}")


def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                      for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = Config.model.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == Config.data.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    prepare_raw_data()
    process_data()
