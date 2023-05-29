import tensorflow as tf
import tokenization
import os
import models.transformer as transformer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


max_tokens = 65
tokenizers = tf.saved_model.load('tokenizers')
tokenizer_eng = tokenizers.eng
tokenizer_ita = tokenizers.ita

num_layers = 6  # encoder-decoder layers
d_model = 128  # size of the key, value, query vector in the attention mechanism
dff = 512  # hidden nodes in the dense mlp
num_heads = 8
epochs = 25
eng_vocab_size = tokenizers.eng.get_vocab_size().numpy()
ita_vocab_size = tokenizers.ita.get_vocab_size().numpy()


class EngItaTranslator(tf.Module):
    """
    english -> italian
    """
    def __init__(self, transformer, tokenizers):
        self.transformer = transformer
        self.tokenizers = tokenizers

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence, max_length=max_tokens):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.eng.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # get [START] and [END] tokens
        start_end = self.tokenizers.ita.tokenize([''])[0]  # -> [[START], , [END]]
        start = start_end[0][tf.newaxis]  # [START] token
        end = start_end[1][tf.newaxis]  # [END] token

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)  # put [START] at the beginning

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # Select the last token from the `seq_len` dimension.
            # without the ':' after -1 the shape would be (batch_size, vocab_size)
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        text = self.tokenizers.ita.detokenize(output)[0]  # Shape: `()`.

        tokens = self.tokenizers.ita.lookup(output)[0]

        return text, tokens


class ItaEngTranslator(tf.Module):
    """
    italian -> english
    """
    def __init__(self, transformer, tokenizers):
        self.transformer = transformer
        self.tokenizers = tokenizers

    def __call__(self, sentence, max_length=max_tokens):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.ita.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # get [START] and [END] tokens
        start_end = self.tokenizers.eng.tokenize([''])[0]  # -> [[START], , [END]]
        start = start_end[0][tf.newaxis]  # [START] token
        end = start_end[1][tf.newaxis]  # [END] token

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)  # put [START] at the beginning

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # Select the last token from the `seq_len` dimension.
            # without the ':' after -1 the shape would be (batch_size, vocab_size)
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        text = self.tokenizers.eng.detokenize(output)[0]  # Shape: `()`.

        tokens = self.tokenizers.eng.lookup(output)[0]

        return text, tokens


class ExportTranslator(tf.Module):
    """
    returns only the translation
    """
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        result, tokens = self.translator(sentence)

        return result
