from __future__ import print_function


from hbconfig import Config
import tensorflow as tf
from tensorflow.contrib import layers



class TextCNN:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32

        self.mode = mode
        self.params = params

        self._init_placeholder(features, labels)
        self.build_graph()

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"prediction": self.prediction})
        else:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=self.train_pred,
                loss=self.loss,
                train_op=self.train_op,
                eval_metric_ops={
                    "accuracy": tf.metrics.accuracy(tf.argmax(self.targets, axis=1), self.predictions)
                }
            )

    def _init_placeholder(self, features, labels):
        self.input_data = features
        if type(features) == dict:
            self.input_data = features["input_data"]

        self.targets = labels

    def build_graph(self):
        self._build_embed()
        self._build_conv_layers()
        self._build_fully_connected_layers()

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss()
            self._build_optimizer()

    def _build_embed(self):
        with tf.variable_scope("embeddings", dtype=self.dtype) as scope:
            embed_type = Config.model.embed_type
            if embed_type == "rand":
                self.embedding = tf.get_variable(
                        "embedding-rand",
                        [Config.data.vocab_size, Config.model.embed_dim],
                        self.dtype)
            elif embed_type == "static":
                raise NotImplementedError("CNN-static not implemented yet.")
            elif embed_type == "non-static":
                raise NotImplementedError("CNN-non-static not implemented yet.")
            elif embed_type == "multichannel":
                raise NotImplementedError("CNN-multichannel not implemented yet.")
            else:
                raise ValueError(f"Unknown embed_type {Config.model.embed_type}")

            self.embedding_input = tf.nn.embedding_lookup(
                self.embedding, self.input_data)
            self.embedding_input_expanded = tf.expand_dims(self.embedding_input, -1)

    def _build_conv_layers(self):
        with tf.variable_scope("convolutions", dtype=self.dtype) as scope:
            pooled_outputs = self._build_conv_maxpool()

            num_total_filters = Config.model.num_filters * len(Config.model.filter_sizes)
            self.concat_pooled = tf.concat(pooled_outputs, 3)
            self.flat_pooled = tf.reshape(self.concat_pooled, [-1, num_total_filters])

            if self.mode == tf.estimator.ModeKeys.TRAIN:
                self.h_dropout = tf.layers.dropout(self.flat_pooled, Config.model.dropout)
            else:
                self.h_dropout = tf.layers.dropout(self.flat_pooled, 0)

    def _build_conv_maxpool(self):
        pooled_outputs = []
        for filter_size in Config.model.filter_sizes:
            with tf.variable_scope(f"conv-maxpool-{filter_size}-filter"):
                conv = tf.layers.conv2d(
                        self.embedding_input_expanded,
                        Config.model.num_filters,
                        (filter_size, Config.model.embed_dim),
                        activation=tf.nn.relu)

                pool = tf.layers.max_pooling2d(
                        conv,
                        (Config.data.max_seq_length - filter_size + 1, 1),
                        (1, 1))

                pooled_outputs.append(pool)
        return pooled_outputs

    def _build_fully_connected_layers(self):
        with tf.variable_scope("fully-connected", dtype=self.dtype) as scope:
            self.output = tf.layers.dense(
                           self.h_dropout,
                           Config.data.num_classes,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())

    def _build_loss(self):
        self.loss = tf.losses.softmax_cross_entropy(
                self.targets,
                self.output,
                scope="loss")

        self.train_pred = tf.argmax(self.output[0], name='train/pred_0')
        self.predictions = tf.argmax(self.output, axis=1)

    def _build_optimizer(self):
        self.train_op = layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'learning_rate'],
            name="train_op")
