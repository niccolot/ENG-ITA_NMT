import tensorflow as tf
import sub_units


class EncoderLayer(tf.keras.layers.Layer):
    """
    single block of encoder model, to be stacked multiple
    time and fed with the positional encoding of the language 1
    """
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = sub_units.GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = sub_units.FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    """
    single block of decoder model, to be stacked multiple
    times and fed with the decoder output as context and
    positional encoding of the inputs of language 2
    """
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super().__init__()

        self.causal_self_attention = sub_units.CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = sub_units.CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = sub_units.FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.

        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = sub_units.PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [EncoderLayer(d_model=d_model,
                                        num_heads=num_heads,
                                        dff=dff, dropout_rate=dropout_rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
                 dropout_rate=0.1):

        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = sub_units.PositionalEmbedding(vocab_size=vocab_size,
                                                           d_model=d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [DecoderLayer(d_model=d_model,
                                        num_heads=num_heads,
                                        dff=dff,
                                        dropout_rate=dropout_rate)
                           for _ in range(num_layers)]

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        return x  # (batch_size, target_seq_len, d_model)
