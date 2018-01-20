
from hbconfig import Config
import tensorflow as tf



class Graph:

    def __init__(self, mode, dtype=tf.float32):
        self.mode = mode
        self.dtype = dtype

    def build(self, input_data):
        embedding_input = self.build_embed(input_data)
        conv_output = self.build_conv_layers(embedding_input)
        return self.build_fully_connected_layers(conv_output)

    def build_embed(self, input_data):
        with tf.variable_scope("embeddings", dtype=self.dtype) as scope:
            embed_type = Config.model.embed_type

            if embed_type == "rand":
                embedding = tf.get_variable(
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
                raise ValueError(f"Unknown embed_type {self.embed_type}")

            return tf.expand_dims(tf.nn.embedding_lookup(embedding, input_data), -1)

    def build_conv_layers(self, embedding_input):
        with tf.variable_scope("convolutions", dtype=self.dtype) as scope:
            pooled_outputs = self._build_conv_maxpool(embedding_input)

            num_total_filters = Config.model.num_filters * len(Config.model.filter_sizes)
            concat_pooled = tf.concat(pooled_outputs, 3)
            flat_pooled = tf.reshape(concat_pooled, [-1, num_total_filters])

            if self.mode == tf.estimator.ModeKeys.TRAIN:
                h_dropout = tf.layers.dropout(flat_pooled, Config.model.dropout)
            else:
                h_dropout = tf.layers.dropout(flat_pooled, 0)
            return h_dropout

    def _build_conv_maxpool(self, embedding_input):
        pooled_outputs = []
        for filter_size in Config.model.filter_sizes:
            with tf.variable_scope(f"conv-maxpool-{filter_size}-filter"):
                conv = tf.layers.conv2d(
                        embedding_input,
                        Config.model.num_filters,
                        (filter_size, Config.model.embed_dim),
                        activation=tf.nn.relu)

                pool = tf.layers.max_pooling2d(
                        conv,
                        (Config.data.max_seq_length - filter_size + 1, 1),
                        (1, 1))

                pooled_outputs.append(pool)
        return pooled_outputs

    def build_fully_connected_layers(self, conv_output):
        with tf.variable_scope("fully-connected", dtype=self.dtype) as scope:
            return tf.layers.dense(
                    conv_output,
                    Config.data.num_classes,
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
