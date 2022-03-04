import re
import sys
import time
import pandas as pd
from gensim.models import KeyedVectors
import gensim
import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, Dropout, GlobalAveragePooling1D, BatchNormalization
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers, optimizers
import pickle
np.random.seed(1003)


def read_labels(Path):
    with open(Path, 'r') as f:
        labels = f.readlines()
    f.close()
    list_of_char = ['\n', '\'']
    pattern = '[' + ''.join(list_of_char) + ']'
    new_labels = [re.sub(pattern, '', i) for i in labels]
    new_labels = [1 if i == 'pos' else 0 for i in new_labels]

    return new_labels


def read_file(Path):
    with open(Path, 'r') as f:
        content = f.readlines()
    f.close()

    return content


def process_list(listOfSentences):
    list_of_char = ['"', '\n', '\'', ',', '.']
    pattern = '[' + ''.join(list_of_char) + ']'
    for i in range(len(listOfSentences)):
        listOfSentences[i] = ' '.join(
            re.sub(pattern, ' ', listOfSentences[i]).split())

    return listOfSentences


def readInData(Path_To_Folder):
    PathToTrain_W_Stopwords = Path_To_Folder + "\\train.csv"
    PathToTrain_WO_Stopwords = Path_To_Folder + "\\train_ns.csv"
    PathToVal_W_Stopwords = Path_To_Folder + "\\val.csv"
    PathToVal_WO_Stopwords = Path_To_Folder + "\\val_ns.csv"
    PathToTest_W_Stopwords = Path_To_Folder + "\\test.csv"
    PathToTest_WO_Stopwords = Path_To_Folder + "\\test_ns.csv"
    PathTolabels = Path_To_Folder + "\\labels.txt"

    xtrain = process_list(read_file(PathToTrain_W_Stopwords))
    ytrain = process_list(read_file(PathToTrain_WO_Stopwords))
    xval = process_list(read_file(PathToVal_W_Stopwords))
    yval = process_list(read_file(PathToVal_WO_Stopwords))
    xtest = process_list(read_file(PathToTest_W_Stopwords))
    ytest = process_list(read_file(PathToTest_WO_Stopwords))
    labels = read_labels(PathTolabels)

    train_labels = labels[:len(xtrain)]
    val_labels = labels[len(xtrain):len(xtrain)+len(xval)]
    test_labels = labels[len(xtrain)+len(xval):]

    return xtrain, ytrain, xval, yval, xtest, ytest, train_labels, val_labels, test_labels


def get_w2vmodel():
    Path_of_w2vec = "a3//data//word2vec.wordvectors"
    wv = KeyedVectors.load(Path_of_w2vec, mmap='r')
    # wv = KeyedVectors.load_word2vec_format(Path_of_w2vec, binary=True)

    return wv


def pre_process(df, MAX_VOCAB_SIZE, MAX_SENT_LEN):
    df_sentence = df.sample(frac=1, random_state=10)
    df_sentence.reset_index(inplace=True, drop=True)
    df_sentence.dropna()

    word_seq = [text_to_word_sequence(sent)
                for sent in df_sentence['sentence']]
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts([' '.join(seq[:MAX_SENT_LEN])
                           for seq in word_seq])
    train = tokenizer.texts_to_sequences(
        [' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq])
    train = pad_sequences(train, maxlen=MAX_SENT_LEN,
                          padding='post', truncating='post')

    df_sentence["polarity"] = pd.to_numeric(df_sentence["polarity"])
    polarity = df_sentence['polarity']

    return train, polarity, tokenizer


def get_embeddingMatrix(embeddings, X_train_tokenizer, EMBEDDING_DIM):
    embeddings_matrix = np.random.uniform(-0.05, 0.05,
                                          size=(len(X_train_tokenizer.word_index)+1, EMBEDDING_DIM))

    for word, i in X_train_tokenizer.word_index.items():
        try:
            embeddings_vector = embeddings[word]
        except KeyError:
            embeddings_vector = None
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector

    return embeddings_matrix


def build_model(ACTIVATION, embeddings_matrix, X_train_tokenizer, X_train, X_train_polarity, X_val, X_val_polarity, EMBEDDING_DIM, ACTIVATION_DIM, BATCH_SIZE, N_EPOCHS):
    x_train_model = Sequential()

    # Input Layer
    x_train_model.add(Embedding(input_dim=len(X_train_tokenizer.word_index)+1,
                                output_dim=EMBEDDING_DIM,
                                weights=[embeddings_matrix], trainable=False, name='word_embedding_layer',
                                mask_zero=True))
    x_train_model.add(GlobalAveragePooling1D())

    # Hidden Layer
    x_train_model.add(
        Dense(50, activation='relu', name=ACTIVATION+'_layer',
              kernel_regularizer=regularizers.l2(0.001)))

    x_train_model.add(Dropout(rate=0.1, name='dropout_1'))

    # Output Layer
    x_train_model.add(
        Dense(1, activation='softmax', name='output_layer'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    x_train_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          optimizer='adam',
                          metrics=['accuracy'])

    x_train_model.fit(X_train, X_train_polarity,
                      batch_size=BATCH_SIZE,
                      epochs=N_EPOCHS,
                      validation_data=(X_val, X_val_polarity),
                      shuffle=True)

    return x_train_model


def main(Path_To_data):

    # These are some hyperparameters that can be tuned
    MAX_SENT_LEN = 40
    BATCH_SIZE = 500
    N_EPOCHS = 10
    MAX_VOCAB_SIZE = 20000
    EMBEDDING_DIM = 100
    RELU_DIM = MAX_SENT_LEN+2
    SIGMOID_DIM = MAX_SENT_LEN
    TANH_DIM = MAX_SENT_LEN

    # Get relevant data from files
    xtrain, ytrain, xval, yval, xtest, ytest, train_labels, val_labels, test_labels = readInData(
        Path_To_data)

    df_train = pd.DataFrame(np.column_stack([xtrain, train_labels]), columns=[
        'sentence', 'polarity'])
    df_val = pd.DataFrame(np.column_stack([xval, val_labels]), columns=[
        'sentence', 'polarity'])

    df_test = pd.DataFrame(np.column_stack([xtest, test_labels]), columns=[
        'sentence', 'polarity'])

    X_train, X_train_polarity, X_train_tokenizer = pre_process(
        df_train, MAX_VOCAB_SIZE, MAX_SENT_LEN)
    X_val, X_val_polarity, X_val_tokenizer = pre_process(
        df_val, MAX_VOCAB_SIZE, MAX_SENT_LEN)
    X_test, X_test_polarity, X_test_tokenizer = pre_process(
        df_test, MAX_VOCAB_SIZE, MAX_SENT_LEN)

    # Loading the word vectors into gensim
    embeddings = get_w2vmodel()
    embeddings_matrix = get_embeddingMatrix(
        embeddings, X_train_tokenizer, EMBEDDING_DIM)

    # Build a sequential model
    X_RELU_model = build_model('relu', embeddings_matrix, X_train_tokenizer, X_train,
                               X_train_polarity, X_val, X_val_polarity, EMBEDDING_DIM, RELU_DIM, BATCH_SIZE, N_EPOCHS)
    X_SIGMOID_model = build_model('sigmoid', embeddings_matrix, X_train_tokenizer, X_train,
                                  X_train_polarity, X_val, X_val_polarity, EMBEDDING_DIM, SIGMOID_DIM, BATCH_SIZE, N_EPOCHS)
    X_TANH_model = build_model('tanh', embeddings_matrix, X_train_tokenizer, X_train,
                               X_train_polarity, X_val, X_val_polarity, EMBEDDING_DIM, TANH_DIM, BATCH_SIZE, N_EPOCHS)
    print(X_RELU_model.summary())
    score1, acc1 = X_RELU_model.evaluate(X_test, X_test_polarity,
                                         batch_size=BATCH_SIZE)

    score2, acc2 = X_SIGMOID_model.evaluate(X_test, X_test_polarity,
                                            batch_size=BATCH_SIZE)

    score3, acc3 = X_TANH_model.evaluate(X_test, X_test_polarity,
                                         batch_size=BATCH_SIZE)

    print("Accuracy on Test Set RELU = {0:4.3f}".format(acc1))
    print("Accuracy on Test Set SIGMOID = {0:4.3f}".format(acc2))
    print("Accuracy on Test Set TANH= {0:4.3f}".format(acc3))

    X_RELU_model.save('a4//data//nn_relu.model')
    X_SIGMOID_model.save('a4//data//nn_sigmoid.model')
    X_TANH_model.save('a4//data//nn_tanh.model')

    with open('a4//data//tokenizer.pickle', 'wb') as handle:
        pickle.dump(X_train_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            f"only one argument ({sys.argv[0]}) was read. Please enter 'py .\a4\main.py C:\\Path\\to\\Data_Folder'.\nSESSION TERMINATED.")
        sys.exit(0)
    else:
        start_time = time.time()
        main(sys.argv[1])
        print("COMPLETED IN --- %s seconds ---" % (time.time() - start_time))
    # py .\a4\main.py C:\\Users\\Ruhol\\Desktop\\4BWinter2022\\MSCI598\\msci-nlp-w22\\a1\\data
