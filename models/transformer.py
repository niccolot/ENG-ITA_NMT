import tensorflow as tf
from models import encoder_decoder


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = encoder_decoder.Encoder(num_layers=num_layers, d_model=d_model,
                                               num_heads=num_heads, dff=dff,
                                               vocab_size=input_vocab_size,
                                               dropout_rate=dropout_rate)

        self.decoder = encoder_decoder.Decoder(num_layers=num_layers, d_model=d_model,
                                               num_heads=num_heads, dff=dff,
                                               vocab_size=target_vocab_size,
                                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        return logits

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'encoder': self.encoder,
            'decoder': self.decoder,
            'final_layer': self.final_layer
        })

        return config
