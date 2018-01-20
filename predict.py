
#-*- coding: utf-8 -*-

import argparse
import os
import sys

from hbconfig import Config
import numpy as np
import tensorflow as tf

import data_loader
from model import Model



def predict(ids):

    X = np.array(data_loader._pad_input(ids, Config.data.max_seq_length), dtype=np.int32)
    X = np.reshape(X, (1, Config.data.max_seq_length))

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"input_data": X},
            num_epochs=1,
            shuffle=False)

    estimator = _make_estimator()
    result = estimator.predict(input_fn=predict_input_fn)

    prediction = next(result)["prediction"]
    return prediction


def _make_estimator():
    params = tf.contrib.training.HParams(**Config.model.to_dict())
    # Using CPU
    run_config = tf.contrib.learn.RunConfig(
        model_dir=Config.train.model_dir,
        session_config=tf.ConfigProto(
            device_count={'GPU': 0}
        ))

    model = Model()
    return tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=Config.train.model_dir,
            params=params,
            config=run_config)


def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


def main():
    data_loader.set_max_seq_length(['train_X_ids', 'test_X_ids'])
    vocab = data_loader.load_vocab("vocab")
    Config.data.vocab_size = len(vocab)

    print("Typing anything :) \n")

    while True:
        sentence = _get_user_input()
        ids = data_loader.sentence2id(vocab, sentence)

        if len(ids) > Config.data.max_seq_length:
            print(f"Max length I can handle is: {Config.data.max_seq_length}")
            continue

        result = predict(ids)
        print(result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.model.batch_size = 1

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    main()
