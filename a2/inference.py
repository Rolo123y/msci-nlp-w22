import os
import pickle
import sys
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

p = os.path.abspath('.')
sys.path.insert(1, p)

import a1.main as m1

def read_txtfile(txtfile):
    with open(txtfile, 'r') as f:
        content = f.readlines()
    f.close()

    content = m1.remove_patterns(content)
    
    return content

def process_list(listOfsentences):

    content = m1.remove_patterns(listOfsentences)
    
    return content

def read_Classifier(classifier_type):
    with open("a2\\data\\" + classifier_type +'.pkl', 'rb') as f:
        classifier = pickle.load(f)
    f.close()

    return classifier

def read_vectorizer(classifier_type):
    with open("a2\\data\\vectorizer_" + classifier_type +'.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    f.close()

    return vectorizer

def read_tfidfTransformer(classifier_type):
    with open("a2\\data\\tfidf_transformer_" + classifier_type +'.pkl', 'rb') as f:
        tfidfTransformer = pickle.load(f)
    f.close()

    return tfidfTransformer

def evaluate(listOfsentences, vectorizer, tdidf_transformer, classifier):

    term_matrix = vectorizer.transform(listOfsentences)
    tranformed_matrix = tdidf_transformer.transform(term_matrix)

    return classifier.predict(tranformed_matrix)

def run(txtfile, classifier_type):
    Original_listOfsentences = read_txtfile(txtfile)
    listOfsentences_W_Stopwords = process_list(Original_listOfsentences)
    tokenized_sents = [word_tokenize(i) for i in listOfsentences_W_Stopwords]
    tokens_without_sw = m1.remove_Stopwords(tokenized_sents, stopwords.words())
    listOfsentences_WO_Stopwords = [(" ").join(sentence) for sentence in tokens_without_sw]
    
    if classifier_type == "mnb_uni":        
        vectorizer = read_vectorizer(classifier_type)
        tdidf_transformer = read_tfidfTransformer(classifier_type)
        classifier = read_Classifier(classifier_type)
        prediction = evaluate(listOfsentences_W_Stopwords, vectorizer,tdidf_transformer, classifier)

        for i in range(len(Original_listOfsentences)):
            print(f"{Original_listOfsentences[i]:150s} {prediction[i]:>3}")
        return 1
    elif classifier_type == "mnb_bi":
        vectorizer = read_vectorizer(classifier_type)
        tdidf_transformer = read_tfidfTransformer(classifier_type)
        classifier = read_Classifier(classifier_type)
        prediction = evaluate(listOfsentences_W_Stopwords, vectorizer,tdidf_transformer, classifier)

        for i in range(len(Original_listOfsentences)):
            print(f"{Original_listOfsentences[i]:150s} {prediction[i]:>3}")

        return 1
    elif classifier_type == "mnb_uni_bi":
        vectorizer = read_vectorizer(classifier_type)
        tdidf_transformer = read_tfidfTransformer(classifier_type)
        classifier = read_Classifier(classifier_type)
        prediction = evaluate(listOfsentences_W_Stopwords, vectorizer,tdidf_transformer, classifier)

        for i in range(len(Original_listOfsentences)):
            print(f"{Original_listOfsentences[i]:150s} {prediction[i]:>3}")

        return 1
    elif classifier_type == "mnb_uni_ns":
        vectorizer = read_vectorizer(classifier_type)
        tdidf_transformer = read_tfidfTransformer(classifier_type)
        classifier = read_Classifier(classifier_type)
        prediction = evaluate(listOfsentences_WO_Stopwords, vectorizer,tdidf_transformer, classifier)

        for i in range(len(Original_listOfsentences)):
            print(f"{Original_listOfsentences[i]:150s} {prediction[i]:>3}")

        return 1
    elif classifier_type == "mnb_bi_ns":
        vectorizer = read_vectorizer(classifier_type)
        tdidf_transformer = read_tfidfTransformer(classifier_type)
        classifier = read_Classifier(classifier_type)
        prediction = evaluate(listOfsentences_WO_Stopwords, vectorizer,tdidf_transformer, classifier)

        for i in range(len(Original_listOfsentences)):
            print(f"{Original_listOfsentences[i]:150s} {prediction[i]:>3}")

        return 1
    elif classifier_type == "mnb_uni_bi_ns":
        vectorizer = read_vectorizer(classifier_type)
        tdidf_transformer = read_tfidfTransformer(classifier_type)
        classifier = read_Classifier(classifier_type)
        prediction = evaluate(listOfsentences_WO_Stopwords, vectorizer,tdidf_transformer, classifier)

        for i in range(len(Original_listOfsentences)):
            print(f"{Original_listOfsentences[i]:150s} {prediction[i]:>3}")

        return 1
    else:
        print("the type of classifier to use is not valid. Please try again with a correct classifier!")
        return -1

if __name__=="__main__":
    if len(sys.argv) < 3:
        print(f"only {len(sys.argv)} arguments provided. Please enter 'py .\a2\inference.py C:\\Path\\to\\afile.txt classifier_to_use'.\nSESSION TERMINATED.")
        sys.exit(0)
    else:
        start_time = time.time()
        run(sys.argv[1], sys.argv[2])
        print("\nCOMPLETED IN --- %s seconds ---" % (time.time() - start_time))
# py .\a2\inference.py C:\\Users\\Ruhol\\Desktop\\4BWinter2022\\MSCI598\\msci-nlp-w22\\a2\\data\\someSentences.txt mnb_uni