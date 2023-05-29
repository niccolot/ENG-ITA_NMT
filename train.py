import tensorflow as tf
import tokenization
import os
import utils
import prepro
import models.transformer as transformer
import export
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


num_layers = 6  # encoder-decoder layers
d_model = 128  # size of the key, value, query vector in the attention mechanism
dff = 512  # hidden nodes in the dense mlp
num_heads = 8
epochs = 20


# custom lr scheduler used in the attention article
learning_rate = utils.CustomSchedule(d_model=d_model)
# adam hyperparameter used in the attention article
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# tokenizers and vocabularies
tokenizers = tf.saved_model.load('tokenizers')
tokenizer_eng = tokenizers.eng.tokenize
tokenizer_ita = tokenizers.ita.tokenize
eng_vocab_size = tokenizers.eng.get_vocab_size().numpy()
ita_vocab_size = tokenizers.ita.get_vocab_size().numpy()

# data loading and processing
ds_train, ds_val = prepro.get_datasets('data/ita_eng_dataset.txt')
train_batches = prepro.get_batches(ds_train, tokenizer_eng, tokenizer_ita)
val_batches = prepro.get_batches(ds_val, tokenizer_eng, tokenizer_ita)

# training
masked_loss = utils.masked_loss
masked_accuracy = utils.masked_accuracy


model = transformer.Transformer(num_layers=num_layers,
                                d_model=d_model,
                                num_heads=num_heads,
                                dff=dff,
                                input_vocab_size=eng_vocab_size,
                                target_vocab_size=ita_vocab_size)

model.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])

history = model.fit(train_batches, epochs=epochs, validation_data=val_batches)

# plotting
acc = history.history['masked_accuracy']
val_acc = history.history['val_masked_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(loss))

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

# save the exportable model
trans = export.ItaEngTranslator(model, tokenizers)
translator = export.ExportTranslator(trans)
tf.saved_model.save(translator, export_dir='translator_eng_ita')
