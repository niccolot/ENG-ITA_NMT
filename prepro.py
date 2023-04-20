import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow_text as tf_text
import re
import pathlib
import numpy as np


def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)


def build_vocabs_from_dataset(lang1_dataset,
                              lang2_dataset,
                              bert_vocab_args,
                              save_vocabs=False,
                              vocab1_path=None,
                              vocab2_path=None):
    """
    given 2 tf Datasets objects with paired translated sentences returns the
    bert vocabularies

    :param lang1_dataset: (tf.data.Dataset) first language dataset
    :param lang2_dataset: (tf.data.Dataset) second language dataset
    :param bert_vocab_args: (dict) arguments for the bert algorithm
    :param save_vocabs: (bool) if one wants to save the vocabularies
    :param vocab1_path: (str) path and name of the .txt file to be saved
    :param vocab2_path: (str) path and name of the .txt file to be saved

    :return: vocabularies
    """

    vocab1 = bert_vocab.bert_vocab_from_dataset(lang1_dataset, **bert_vocab_args)
    vocab2 = bert_vocab.bert_vocab_from_dataset(lang2_dataset, **bert_vocab_args)

    if save_vocabs:
        write_vocab_file(vocab1_path, vocab1)
        write_vocab_file(vocab2_path, vocab2)

    return vocab1, vocab2


def get_datasets(dataset):
    """
    given a .txt file with translations separated by '\t' returns 2 separate
    tensorflow datasets of one language each

    input: dataset (str): path to the dataset file
    return:
    """

    dataset1, dataset2 = np.loadtxt(dataset,
                                    usecols=(0, 1),
                                    encoding='utf-8',
                                    unpack=True,
                                    dtype='str',
                                    delimiter='\t')

    dataset1 = tf.convert_to_tensor(dataset1)
    dataset2 = tf.convert_to_tensor(dataset2)

    dataset1 = tf.data.Dataset.from_tensor_slices(dataset1)
    dataset2 = tf.data.Dataset.from_tensor_slices(dataset2)

    return dataset1, dataset2


class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        super().__init__()
        self.tokenizer = tf_text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        # makes the export work when the model needs an external file
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        # Create the signatures for export:

        # Include a 'tokenize' signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    def _add_start_end(self, ragged):
        """
        adds the 'start' and 'end' tokens at the beginning and
        end of each sentence. This will come handy when the tokenized
        text will be fed to the transformer

        :param ragged: input dataset
        :return: dataset with 'start' and 'end' tokens
        """
        start_tk = tf.argmax(tf.constant(self._reserved_tokens) == "[START]")
        end_tk = tf.argmax(tf.constant(self._reserved_tokens) == "[END]")
        count = ragged.bounding_shape()[0]
        starts = tf.fill([count, 1], start_tk)
        ends = tf.fill([count, 1], end_tk)

        return tf.concat([starts, ragged, ends], axis=1)

    def _cleanup_text(self, token_txt):
        """
        removes all but the 'unknown' tokens from the detokenized text
        before turning it back into words.
        It also joins the text with white
        spaces, otherwise the output would be like [b'i', b'am'] and not
        [b'i am']

        :param token_txt: detokenized text to be tidied up
        :return: readable detokenized text
        """
        # Drop the reserved tokens, except for "[UNK]".
        bad_tokens = [re.escape(tok) for tok in self._reserved_tokens if tok != "[UNK]"]
        bad_token_re = "|".join(bad_tokens)

        bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
        result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

        # Join them into strings.
        result = tf.strings.reduce_join(result, separator=' ', axis=-1)

        return result

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        # adds the 'start' and 'end' tokens
        enc = self._add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        # makes the detokenized text readable
        return self._cleanup_text(words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)
