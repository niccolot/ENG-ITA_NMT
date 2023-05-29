import tensorflow as tf
import os
import export
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('\nPress 1 for english to italian, press 2 for italian to english')
choice = int(input())

if choice == 1:
    translator = tf.saved_model.load('translator_eng_ita')

if choice == 2:
    translator = tf.saved_model.load('translator_ita_eng')

if choice != 1 and choice != 2:
    raise Exception("Invalid input")

print('\nType the sentence to be translated')
sentence = input()

print('\nTranslation:')
print(translator(sentence).numpy().decode("utf-8"))
