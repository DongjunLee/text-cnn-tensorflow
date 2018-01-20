from __future__ import print_function


from hbconfig import Config
import tensorflow as tf
from tensorflow.contrib import layers

import text_cnn



class Model:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32

        self.mode = mode
        self.params = params

        self.loss, self.train_op, self.metrics, self.predictions = None, None, None, None
        self._init_placeholder(features, labels)
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions={"prediction": self.predictions})

    def _init_placeholder(self, features, labels):
        self.input_data = features
        if type(features) == dict:
            self.input_data = features["input_data"]

        self.targets = labels

    def build_graph(self):
        graph = text_cnn.Graph(self.mode)
        output = graph.build(self.input_data)

        self._build_prediction(output)
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(output)
            self._build_optimizer()
            self._build_metric()

    def _build_loss(self, output):
        self.loss = tf.losses.softmax_cross_entropy(
                self.targets,
                output,
                scope="loss")

    def _build_prediction(self, output):
        tf.argmax(output[0], name='train/pred_0') # for print_verbose
        self.predictions = tf.argmax(output, axis=1)

    def _build_optimizer(self):
        self.train_op = layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'learning_rate'],
            name="train_op")

    def _build_metric(self):
        self.metrics = {
            "accuracy": tf.metrics.accuracy(tf.argmax(self.targets, axis=1), self.predictions)
        }
