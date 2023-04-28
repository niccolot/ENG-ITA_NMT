import tensorflow as tf

# hyper_parameters
batch_size = 64
max_tokens = 65
buffer_size = 20000  # big number for the shuffling function


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

    lang2 = tokenizer_lang1(lang2)

    # for input and labels the last and first token respectively will
    # be dropped for the teacher forcing, so the +1
    lang2 = lang2[:, :max_tokens+1]
    lang2_input = lang2[:, :-1].to_tensor()  # drop last token
    lang2_target = lang2[:, 1:].to_tensor()  # drop first token

    return (lang1, lang2_input), lang2_target


def get_batches(dataset):
    """
    get the preprocessed dataset to be fed to the model in a teacher forcing fashion

    :param dataset: (lang1, lang2_input), lang2_target teacher forcing dataset
    :return: preprocessed dataset
    """
    return dataset\
        .shuffle(buffer_size)\
        .batch(batch_size)\
        .map(teacher_forcing, tf.data.AUTOTUNE)\
        .prefetch(buffer_size=tf.data.AUTOTUNE)
