
from hbconfig import Config
import tensorflow as tf

import data_loader
import dataset
from model import TextCNN
import hook



def experiment_fn(run_config, params):

    text_cnn = TextCNN()
    estimator = tf.estimator.Estimator(
            model_fn=text_cnn.model_fn,
            model_dir=Config.train.model_dir,
            params=params,
            config=run_config)

    vocab = data_loader.load_vocab("vocab")
    Config.data.vocab_size = len(vocab)

    train_X, test_X, train_y, test_y = data_loader.make_train_and_test_set()

    train_input_fn, train_input_hook = dataset.get_train_inputs(train_X, train_y)
    test_input_fn, test_input_hook = dataset.get_test_inputs(test_X, test_y)

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        train_steps=Config.train.train_steps,
        min_eval_frequency=Config.train.min_eval_frequency,
        train_monitors=[
            train_input_hook,
            hook.print_input(
                variables=['train/input_0'],
                vocab=vocab,
                every_n_iter=Config.train.check_hook_n_iter),
            hook.print_target(
                variables=['train/target_0', 'train/pred_0'],
                every_n_iter=Config.train.check_hook_n_iter)],
        eval_hooks=[test_input_hook]
    )
    return experiment
