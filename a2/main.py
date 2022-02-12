import os
import sys
import time
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

def read_labels(Path):

    with open(Path, 'r') as f:
        labels = f.readlines()
    f.close()
    list_of_char = ['\n','\'']
    pattern = '[' + ''.join(list_of_char) + ']'
    new_labels = [re.sub(pattern, '', i) for i in labels]

    return new_labels

def read_file(Path):

    with open(Path, 'r') as f:
        content = f.readlines()
    f.close()
    
    return content

def process_list(listOfSentences):

    list_of_char = ['"','\n', '\'', ',', '.']
    pattern = '[' + ''.join(list_of_char) + ']'
    for i in range(len(listOfSentences)):
        listOfSentences[i] = ' '.join(re.sub(pattern, ' ' , listOfSentences[i]).split())

    return listOfSentences

def readInData(Path):

    PathToTrain_W_Stopwords = Path + "\\train.csv"
    PathToTrain_WO_Stopwords = Path + "\\train_ns.csv"
    PathToVal_W_Stopwords = Path + "\\val.csv"
    PathToVal_WO_Stopwords = Path + "\\val_ns.csv"
    PathToTest_W_Stopwords = Path + "\\test.csv"
    PathToTest_WO_Stopwords = Path + "\\test_ns.csv"
    PathTolabels = Path + "\\labels.txt"

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

def write_outto_pkl(Path, filename, classifier, vectorizer, tfidf_transformer):
    fullPathOfClassifier = Path + filename
    fullPathOfvectorizer = Path + "vectorizer_" + filename
    fullPathOftransformer = Path + "tfidf_transformer_" + filename
    if os.path.isfile(fullPathOfClassifier) == True:
        print(f'\t classifier already exists in {fullPathOfClassifier}. Please delete it and try again!')
        return -1
    elif os.path.isfile(fullPathOfvectorizer) == True:
        print(f'\t vectorizer already exists in {fullPathOfvectorizer}. Please delete it and try again!')
        return -1
    elif os.path.isfile(fullPathOftransformer) == True:
        print(f'\t tfidf_transformer already exists in {fullPathOftransformer}. Please delete it and try again!')
        return -1
    else:
        with open(fullPathOfClassifier, 'wb') as f:
            pickle.dump(classifier, f)
        f.close()
        with open(fullPathOfvectorizer, 'wb') as f:
            pickle.dump(vectorizer, f)
        f.close()
        with open(fullPathOftransformer, 'wb') as f:
            pickle.dump(tfidf_transformer, f)
        f.close()
        print(f'\tCreated files named {filename},vectorizer_{filename},tfidf_transformer_{filename} located in folder {Path}')

    return 1

def Train_On_Dataset_for_unigrams(Train_Data, Correct_labels):
    print("\tworking on Unigrams")

    # Vectorize each word in each sentence in the dataset
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,1))
    vectorized = vectorizer.fit_transform(Train_Data)
    print("\t\tvectorized")

    # for each word in a sentence, calculate its importance in the sentence using tf-idf
    Tfidf_T = TfidfTransformer()
    tfidf_transformed = Tfidf_T.fit_transform(vectorized)
    print("\t\ttfidf-transformed")

    # Train the classifier on given dataset and the corresponsing labels
    alpha_param = 1.5
    classifier = MultinomialNB(alpha=alpha_param).fit(tfidf_transformed, Correct_labels)
    print("\t\ttrained on MultinomialNB")

    return classifier, vectorizer, Tfidf_T

def Train_On_Dataset_for_bigrams(Train_Data, Correct_labels):
    print("\tworking on bigrams")

    # Vectorize each word in each sentence in the dataset
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    vectorized = vectorizer.fit_transform(Train_Data)
    print("\t\tvectorized")

    # for each word in a sentence, calculate its importance in the sentence using tf-idf
    Tfidf_T = TfidfTransformer()
    tfidf_transformed = Tfidf_T.fit_transform(vectorized)
    print("\t\ttfidf-transformed")

    # Train the classifier on given dataset and the corresponsing labels
    alpha_param = 1.5
    classifier = MultinomialNB(alpha=alpha_param).fit(tfidf_transformed, Correct_labels)
    print("\t\ttrained on MultinomialNB")

    return classifier, vectorizer, Tfidf_T

def Train_On_Dataset_for_unigramANDbigrams(Train_Data, Correct_labels):
    print("\tworking on unigram AND bigrams")

    # Vectorize each word in each sentence in the dataset
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    vectorized = vectorizer.fit_transform(Train_Data)
    print("\t\tvectorized")

    # for each word in a sentence, calculate its importance in the sentence using tf-idf
    Tfidf_T = TfidfTransformer()
    tfidf_transformed = Tfidf_T.fit_transform(vectorized)
    print("\t\ttfidf-transformed")

    # Train the classifier on given dataset and the corresponsing labels
    alpha_param = 1.5
    classifier = MultinomialNB(alpha=alpha_param).fit(tfidf_transformed, Correct_labels)
    print("\t\ttrained on MultinomialNB")

    return classifier, vectorizer, Tfidf_T

def prediction(classifier, vectorizer, Tfidf_T, listOfSentences, Correct_labels):
    print("\tPredicting ...")

    # Vectorize the words in each sentence
    vect = vectorizer.transform(listOfSentences)
    print("\t\tvectorized")

    # for each word in a sentence, calculate its importance in the sentence using tf-idf
    Tfidf_ = Tfidf_T.transform(vect)
    print("\t\ttfidf-transformed")

    # classifiers prediction against labels
    prediction = classifier.predict(Tfidf_)
    correct_res = (Correct_labels == prediction).sum()
    wrong_res = (Correct_labels != prediction).sum()
    accuracy = classifier.score(Tfidf_, Correct_labels)
    print(f'\t\tPredction Complete: result: {prediction} \t\tAccuracy:{accuracy}')

    return [correct_res, wrong_res, accuracy], classifier, vectorizer, Tfidf_T

def main(Path):

    # Get relevant data from files
    xtrain, ytrain, xval, yval, xtest, ytest, train_labels, val_labels, test_labels = readInData(Path)

    # Train WITH Stopwords for unigrams, bigrams, unigramsANDbigrams
    print("TRAINING ON DATASET WITH STOPWORDS")
    classifier_WStop_unigram, vectorizer_WStop_unigram, Tfidf_T_WStop_unigram = Train_On_Dataset_for_unigrams(xtrain, train_labels)
    classifier_WStop_bigram, vectorizer_WStop_bigram, Tfidf_T_WStop_bigram = Train_On_Dataset_for_bigrams(xtrain, train_labels)
    classifier_WStop_unigramANDbigram, vectorizer_WStop_unigramANDbigram, Tfidf_T_WStop_unigramANDbigram = Train_On_Dataset_for_unigramANDbigrams(xtrain, train_labels)

    # Train WITHOUT Stopwords for unigrams, bigrams, unigramsANDbigrams
    print("TRAINING ON DATASET WITHOUT STOPWORDS")
    classifier_WOStop_unigram, vectorizer_WOStop_unigram, Tfidf_T_WOStop_unigram = Train_On_Dataset_for_unigrams(ytrain, train_labels)
    classifier_WOStop_bigram, vectorizer_WOStop_bigram, Tfidf_T_WOStop_bigram = Train_On_Dataset_for_bigrams(ytrain, train_labels)
    classifier_WOStop_unigramANDbigram, vectorizer_WOStop_unigramANDbigram, Tfidf_T_WOStop_unigramANDbigram = Train_On_Dataset_for_unigramANDbigrams(ytrain, train_labels)
    
    # Validate the classifiers WITH Stopwords and find good alpha values for each MultinomialNB
    print("VALIDATING ON DATASET WITH STOPWORDS")
    Unigram_WStop_Val, classifier_WStop_unigram, vectorizer_WStop_unigram, Tfidf_T_WStop_unigram = prediction(classifier_WStop_unigram, vectorizer_WStop_unigram, Tfidf_T_WStop_unigram, xval, val_labels)
    bigram_WStop_Val, classifier_WStop_bigram, vectorizer_WStop_bigram, Tfidf_T_WStop_bigram = prediction(classifier_WStop_bigram, vectorizer_WStop_bigram, Tfidf_T_WStop_bigram, xval, val_labels)
    unigramANDbigram_WStop_Val, classifier_WStop_unigramANDbigram, vectorizer_WStop_unigramANDbigram, Tfidf_T_WStop_unigramANDbigram = prediction(classifier_WStop_unigramANDbigram, vectorizer_WStop_unigramANDbigram, Tfidf_T_WStop_unigramANDbigram, xval, val_labels)

    # Validate the classifiers WITHOUT Stopwords and find good alpha values for each MultinomialNB
    print("VALIDATING ON DATASET WITHOUT STOPWORDS")
    Unigram_WOStop_Val, classifier_WOStop_unigram, vectorizer_WOStop_unigram, Tfidf_T_WOStop_unigram = prediction(classifier_WOStop_unigram, vectorizer_WOStop_unigram, Tfidf_T_WOStop_unigram, yval, val_labels)
    bigram_WOStop_Val, classifier_WOStop_bigram, vectorizer_WOStop_bigram, Tfidf_T_WOStop_bigram = prediction(classifier_WOStop_bigram, vectorizer_WOStop_bigram, Tfidf_T_WOStop_bigram, yval, val_labels)
    unigramANDbigram_WOStop_Val, classifier_WOStop_unigramANDbigram, vectorizer_WOStop_unigramANDbigram, Tfidf_T_WOStop_unigramANDbigram = prediction(classifier_WOStop_unigramANDbigram, vectorizer_WOStop_unigramANDbigram, Tfidf_T_WOStop_unigramANDbigram, yval, val_labels)
    
    print("TESTING ON TEST SET WITH STOPWORDS")
    Unigram_WStop_test, classifier_WStop_unigram, vectorizer_WStop_unigram, Tfidf_T_WStop_unigram = prediction(classifier_WStop_unigram, vectorizer_WStop_unigram, Tfidf_T_WStop_unigram, xtest, test_labels)
    bigram_WStop_test, classifier_WStop_bigram, vectorizer_WStop_bigram, Tfidf_T_WStop_bigram = prediction(classifier_WStop_bigram, vectorizer_WStop_bigram, Tfidf_T_WStop_bigram,xtest, test_labels)
    unigramANDbigram_WStop_test, classifier_WStop_unigramANDbigram, vectorizer_WStop_unigramANDbigram, Tfidf_T_WStop_unigramANDbigram = prediction(classifier_WStop_unigramANDbigram, vectorizer_WStop_unigramANDbigram, Tfidf_T_WStop_unigramANDbigram, xtest, test_labels)

    print("TESTING ON TEST SET WITHOUT STOPWORDS")
    Unigram_WOStop_test, classifier_WOStop_unigram, vectorizer_WOStop_unigram, Tfidf_T_WOStop_unigram = prediction(classifier_WOStop_unigram, vectorizer_WOStop_unigram, Tfidf_T_WOStop_unigram, ytest, test_labels)
    bigram_WOStop_test, classifier_WOStop_bigram, vectorizer_WOStop_bigram, Tfidf_T_WOStop_bigram = prediction(classifier_WOStop_bigram, vectorizer_WOStop_bigram, Tfidf_T_WOStop_bigram, ytest, test_labels)
    unigramANDbigram_WOStop_test, classifier_WOStop_unigramANDbigram, vectorizer_WOStop_unigramANDbigram, Tfidf_T_WOStop_unigramANDbigram = prediction(classifier_WOStop_unigramANDbigram, vectorizer_WOStop_unigramANDbigram, Tfidf_T_WOStop_unigramANDbigram, ytest, test_labels)
    
    print(f'\nUnigram_WStop_test {Unigram_WStop_test}\nbigram_WStop_test {bigram_WStop_test}\nunigramANDbigram_WStop_test {unigramANDbigram_WStop_test}')
    print(f'Unigram_WOStop_test {Unigram_WOStop_test}\nbigram_WOStop_test {bigram_WOStop_test}\nunigramANDbigram_WOStop_test {unigramANDbigram_WOStop_test}')

    print("\nPlease wait while the program creates the output files ...")
    write_outto_pkl("a2\\data\\" , "mnb_uni.pkl", classifier_WStop_unigram, vectorizer_WStop_unigram, Tfidf_T_WStop_unigram)
    write_outto_pkl("a2\\data\\" , "mnb_bi.pkl", classifier_WStop_bigram, vectorizer_WStop_bigram, Tfidf_T_WStop_bigram)
    write_outto_pkl("a2\\data\\" , "mnb_uni_bi.pkl", classifier_WStop_unigramANDbigram, vectorizer_WStop_unigramANDbigram, Tfidf_T_WStop_unigramANDbigram)
    write_outto_pkl("a2\\data\\" , "mnb_uni_ns.pkl", classifier_WOStop_unigram, vectorizer_WOStop_unigram, Tfidf_T_WOStop_unigram)
    write_outto_pkl("a2\\data\\" , "mnb_bi_ns.pkl", classifier_WOStop_bigram, vectorizer_WOStop_bigram, Tfidf_T_WOStop_bigram)
    write_outto_pkl("a2\\data\\" , "mnb_uni_bi_ns.pkl", classifier_WOStop_unigramANDbigram, vectorizer_WOStop_unigramANDbigram, Tfidf_T_WOStop_unigramANDbigram)

    return 1

if __name__=="__main__":
    if len(sys.argv) < 2:
        print(f"only one argument ({sys.argv[0]}) was read. Please enter 'py .\A1\main.py C:\\Path\\to\\Data_Folder'.\nSESSION TERMINATED.")
        sys.exit(0)
    else:
        start_time = time.time()
        main(sys.argv[1])
        print("COMPLETED IN --- %s seconds ---" % (time.time() - start_time))
# py .\a2\main.py C:\\Users\\Ruhol\\Desktop\\4BWinter2022\\MSCI598\\msci-nlp-w22\\a1\\data