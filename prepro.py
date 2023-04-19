import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow_text as tf_text
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    '''
    given 2 tf Datasets objects with paired translated sentences returns the
    bert vocabularies

    :param lang1_dataset: (tf.data.Dataset) first language dataset
    :param lang2_dataset: (tf.data.Dataset) second language dataset
    :param bert_vocab_args: (dict) arguments for the bert algorithm
    :param save_vocabs: (bool) if one wants to save the vocabularies
    :param vocab1_path: (str) path and name of the .txt file to be saved
    :param vocab2_path: (str) path and name of the .txt file to be saved

    :return: vocabularies
    '''

    vocab1 = bert_vocab.bert_vocab_from_dataset(lang1_dataset, **bert_vocab_args)
    vocab2 = bert_vocab.bert_vocab_from_dataset(lang2_dataset, **bert_vocab_args)

    if save_vocabs:
        write_vocab_file(vocab1_path, vocab1)
        write_vocab_file(vocab2_path, vocab2)

    return vocab1, vocab2


def get_datasets(dataset):
    '''
    given a .txt file with translations separated by '\t' returns 2 separate
    tensorflow datasets of one language each

    input: dataset (str): path to the dataset file
    return:
    '''

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


bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
bert_vocab_args = dict(
    vocab_size=15000,
    reserved_tokens=reserved_tokens,
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={},
)

eng_dataset, ita_dataset = get_datasets('ita_eng_dataset.txt')
vocab_ita, vocab_eng = build_vocabs_from_dataset(eng_dataset,
                                                 ita_dataset,
                                                 bert_vocab_args,
                                                 save_vocabs=True,
                                                 vocab1_path='eng_vocab.txt',
                                                 vocab2_path='ita_vocab.txt')

ita_tokenizer = tf_text.BertTokenizer('ita_vocab.txt', **bert_tokenizer_params)
eng_tokenizer = tf_text.BertTokenizer('eng_vocab.txt', **bert_tokenizer_params)





tokens_eng = eng_tokenizer.tokenize(['tiny tina was going to school, she did not have a care in the world.'])
tokens_eng = tokens_eng.merge_dims(-2, -1)
print(tokens_eng)
words_eng = eng_tokenizer.detokenize(tokens_eng)
print(tf.strings.reduce_join(words_eng, separator=' ', axis=-1))

tokens_ita = ita_tokenizer.tokenize(['tiny tina andava a scuola e non aveva pensieri'])
tokens_ita = tokens_ita.merge_dims(-2, -1)
print(tokens_ita)
words_ita = ita_tokenizer.detokenize(tokens_ita)
print(tf.strings.reduce_join(words_ita, separator=' ', axis=-1))


