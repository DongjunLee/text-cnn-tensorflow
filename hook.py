
from hbconfig import Config
import numpy as np
import tensorflow as tf




def print_variables(variables, rev_vocab=None, every_n_iter=100):

    return tf.train.LoggingTensorHook(
        variables,
        every_n_iter=every_n_iter,
        formatter=format_variable(variables, rev_vocab=rev_vocab))


def format_variable(keys, rev_vocab=None):

    def to_str(sequence):
        if type(sequence) == np.ndarray:
            tokens = [
                rev_vocab.get(x, '') for x in sequence if x != Config.data.PAD_ID]
            return ' '.join(tokens)
        else:
            x = int(sequence)
            return rev_vocab[x]

    def format(values):
        result = []
        for key in keys:
            if rev_vocab is None:
                result.append(f"{key} = {values[key]}")
            else:
                result.append(f"{key} = {to_str(values[key])}")

        try:
            return '\n - '.join(result)
        except:
            pass

    return format


def get_rev_vocab(vocab):
    if vocab is None:
        return None
    return {idx: key for key, idx in vocab.items()}


def print_target(variables, every_n_iter=100):

    return tf.train.LoggingTensorHook(
        variables,
        every_n_iter=every_n_iter,
        formatter=print_pos_or_neg(variables))


def print_pos_or_neg(keys):

    def format(values):
        result = []
        for key in keys:
            if type(values[key]) == np.ndarray:
                value = max(values[key])
            else:
                value = values[key]
            result.append(f"{key} = {value}")

        try:
            return ', '.join(result)
        except:
            pass

    return format
