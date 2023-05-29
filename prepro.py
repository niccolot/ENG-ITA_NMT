import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# hyper_parameters
batch_size = 64
max_tokens = 65
buffer_size = 20000  # big number for the shuffling function


def get_datasets(dataset, zipped=True, test_size=0.1):
    """
    given a .txt file with translations separated by '\t' returns 2 separate
    tensorflow datasets of one language each

    input: dataset (str): path to the dataset file
    param: zipped (bool) if one needs the paired translations
    param: test_size (float) % of data in val dataset
    return: separated or paired tf.data.Dataset objects
    """

    dataset1, dataset2 = np.loadtxt(dataset,
                                    usecols=(0, 1),
                                    encoding='utf-8',
                                    unpack=True,
                                    dtype='str',
                                    delimiter='\t')

    ds1_train, ds1_val, ds2_train, ds2_val = train_test_split(dataset1, dataset2, test_size=test_size)

    ds1_train = tf.convert_to_tensor(ds1_train)
    ds1_val = tf.convert_to_tensor(ds1_val)
    ds2_train = tf.convert_to_tensor(ds2_train)
    ds2_val = tf.convert_to_tensor(ds2_val)

    ds1_train = tf.data.Dataset.from_tensor_slices(ds1_train)
    ds1_val = tf.data.Dataset.from_tensor_slices(ds1_val)
    ds2_train = tf.data.Dataset.from_tensor_slices(ds2_train)
    ds2_val = tf.data.Dataset.from_tensor_slices(ds2_val)

    if zipped:
        train_zipped = tf.data.Dataset.zip((ds1_train, ds2_train))
        val_zipped = tf.data.Dataset.zip((ds1_val, ds2_val))
        return train_zipped, val_zipped

    else:
        return ds1_train, ds1_val, ds2_train, ds2_val


def teacher_forcing(lang1, lang2, tokenizer_lang1, tokenizer_lang2):
    """
    returns the data  with the
    'teacher forcing' training, so if one has to
    translate lang1->lang2 ((lang1 input, lang2 input), shifted lang2)
    i.e. the model's input are a sentence in lang1 for the encoder,
    the same sentence in lang2 for the decoder (with causal padding) and
    for the ground truth the sentence in lang2 but shifted of one token.

    In this way the model sees up to a certain token of the sentence in lang2
    and has to predict the next, e.g. :

    lang1: 'i am writing things'
    lang2: 'sto scrivendo cose'

    'i am writing' -> encoder
    'sto scrivendo' -> decoder
    target = 'cose'

    :param lang1: (ragged tensor) batch of sentences in language 1
    :param lang2: (ragged tensor) batch of sentences in language 2
    :param tokenizer_lang1: tokenizer for language 1
    :param tokenizer_lang2: tokenizer for language 2
    :return: teacher forcing paired dataset
    """

    lang1 = tokenizer_lang1(lang1)
    lang1 = lang1[:, :max_tokens]  # trim longer sequences
    lang1 = lang1.to_tensor()  # ragged to 0 padded dense tensor

    lang2 = tokenizer_lang2(lang2)

    # for input and labels the last and first token respectively will
    # be dropped for the teacher forcing, so the +1
    lang2 = lang2[:, :max_tokens+1]
    lang2_input = lang2[:, :-1].to_tensor()  # drop last token
    lang2_target = lang2[:, 1:].to_tensor()  # drop first token

    return (lang1, lang2_input), lang2_target


def get_batches(dataset, tokenizer_lang1, tokenizer_lang2):
    """
    get the preprocessed dataset to be fed to the model in a teacher forcing fashion

    :param dataset: paired dataset (lang1, lang2)
    :param tokenizer_lang1 tokenizer for language 1
    :param tokenizer_lang2 tokenizer for language 2
    :return: preprocessed dataset
    """
    return dataset\
        .shuffle(buffer_size)\
        .batch(batch_size)\
        .map(lambda x, y: teacher_forcing(x, y, tokenizer_lang1, tokenizer_lang2))\
        .prefetch(buffer_size=tf.data.AUTOTUNE)
