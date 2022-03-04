import re
import sys
import time
import pandas as pd
import numpy as np
import pickle
import gensim
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
np.random.seed(1003)


def read_textfile(Path_To_txt):
    with open(Path_To_txt, 'r') as f:
        context = f.readlines()

    list_of_char = ['\n', '\'']
    pattern = '[' + ''.join(list_of_char) + ']'
    listofsentences = [re.sub(pattern, '', i) for i in context]
    return listofsentences


def read_tokenizer():
    with open('a4//data//tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer


def run(Path_To_txt, classifier_type):

    MAX_SENT_LEN = 30
    path_to_model = 'a4//data//nn_'+str(classifier_type)+'.model'
    model = tf.keras.models.load_model(path_to_model)
    tokenizer = read_tokenizer()

    to_be_predicted = read_textfile(Path_To_txt)

    word_seq = [text_to_word_sequence(sent) for sent in to_be_predicted]
    X_predict = tokenizer.texts_to_sequences(
        [' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq])
    X_predict = pad_sequences(
        X_predict, maxlen=MAX_SENT_LEN, padding='post', truncating='post')

    predict_x = model.predict(X_predict)
    classes_x = np.argmax(predict_x, axis=1)

    for i in range(len(to_be_predicted)):
        pos_or_neg = 'pos' if classes_x[i] == 1 else 'neg'
        print(f'{to_be_predicted[i]} \nprediction ->\t {pos_or_neg}\n')

    return 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"only {len(sys.argv)} arguments provided. Please enter 'py .\a4\inference.py C:\\Path\\to\\afile.txt classifier_to_use'\nSESSION TERMINATED.")
        sys.exit(0)
    else:
        start_time = time.time()
        run(sys.argv[1], sys.argv[2])
        print("\nCOMPLETED IN --- %s seconds ---" % (time.time() - start_time))
# py .\a4\inference.py C:\\Users\\Ruhol\\Desktop\\4BWinter2022\\MSCI598\\msci-nlp-w22\\a4\\data\\testInference.txt relu
