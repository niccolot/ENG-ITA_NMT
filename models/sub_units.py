import tensorflow as tf
import numpy as np


def positional_encoding(length, depth):

    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth

    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate([np.sin(angle_rads),
                                  np.cos(angle_rads)],
                                  axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positional_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_mdoel': self.d_model,
            'embedding': self.embedding,
            'pos_encoding': self.pos_encoding
        })

        return config


class BaseAttention(tf.keras.layers.Layer):
    """
    basic attention init method to be inherited by
    the all other attention blocks
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'mha': self.mha,
            'layernorm': self.layernorm,
            'add': self.add
        })

        return config


class GlobalSelfAttention(BaseAttention):
    """
    self attention in the encoder block
    """
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    """
    masked attention for the decoder's first block
    """
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CrossAttention(BaseAttention):
    """
    cross attention block with keys and values coming from
    the encoder and query coming from the output of the masked
    attention in the decoder
    """
    def call(self, x, context):
        attn_output, attn_scores = self.mha(query=x,
                                            key=context,
                                            value=context,
                                            return_attention_scores=True)

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    # feedforward mlp with residual connection
    def __init__(self, d_model, dff, dropout_rate=0.1):
        """
        :param d_model: (int) models' s depth, same as head_size param in multi head attention layer
        :param dff: (int) depth of the ff dense network
        :param dropout_rate: (float) dropout after the mlp, 0.1 in the original paper
        """
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'seq': self.seq,
            'add': self.add,
            'layer_norm': self.layer_norm
        })
        return config
