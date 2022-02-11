import pickle
import sys
import time
import re

def read_txtfile(txtfile):
    with open(txtfile, 'r') as f:
        content = f.readlines()
    f.close()

    for i in range(len(content)):
        content[i] = re.sub(r'\n','', content[i])
    
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
    listOfsentences = read_txtfile(txtfile)
    
    if classifier_type == "mnb_uni":        
        vectorizer = read_vectorizer(classifier_type)
        tdidf_transformer = read_tfidfTransformer(classifier_type)
        classifier = read_Classifier(classifier_type)
        prediction = evaluate(listOfsentences, vectorizer,tdidf_transformer, classifier)

        for i in range(len(listOfsentences)):
            print(f"{listOfsentences[i]:150s} {prediction[i]:>3}")
        return 1
    elif classifier_type == "mnb_bi":
        vectorizer = read_vectorizer(classifier_type)
        tdidf_transformer = read_tfidfTransformer(classifier_type)
        classifier = read_Classifier(classifier_type)
        prediction = evaluate(listOfsentences, vectorizer,tdidf_transformer, classifier)

        for i in range(len(listOfsentences)):
            print(f"{listOfsentences[i]:150s} {prediction[i]:>3}")

        return 1
    elif classifier_type == "mnb_uni_bi":
        vectorizer = read_vectorizer(classifier_type)
        tdidf_transformer = read_tfidfTransformer(classifier_type)
        classifier = read_Classifier(classifier_type)
        prediction = evaluate(listOfsentences, vectorizer,tdidf_transformer, classifier)

        for i in range(len(listOfsentences)):
            print(f"{listOfsentences[i]:150s} {prediction[i]:>3}")

        return 1
    elif classifier_type == "mnb_uni_ns":
        vectorizer = read_vectorizer(classifier_type)
        tdidf_transformer = read_tfidfTransformer(classifier_type)
        classifier = read_Classifier(classifier_type)
        prediction = evaluate(listOfsentences, vectorizer,tdidf_transformer, classifier)

        for i in range(len(listOfsentences)):
            print(f"{listOfsentences[i]:150s} {prediction[i]:>3}")

        return 1
    elif classifier_type == "mnb_bi_ns":
        vectorizer = read_vectorizer(classifier_type)
        tdidf_transformer = read_tfidfTransformer(classifier_type)
        classifier = read_Classifier(classifier_type)
        prediction = evaluate(listOfsentences, vectorizer,tdidf_transformer, classifier)

        for i in range(len(listOfsentences)):
            print(f"{listOfsentences[i]:150s} {prediction[i]:>3}")

        return 1
    elif classifier_type == "mnb_uni_bi_ns":
        vectorizer = read_vectorizer(classifier_type)
        tdidf_transformer = read_tfidfTransformer(classifier_type)
        classifier = read_Classifier(classifier_type)
        prediction = evaluate(listOfsentences, vectorizer,tdidf_transformer, classifier)

        for i in range(len(listOfsentences)):
            print(f"{listOfsentences[i]:150s} {prediction[i]:>3}")

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
        print("COMPLETED IN --- %s seconds ---" % (time.time() - start_time))
# py .\a2\inference.py C:\\Users\\Ruhol\\Desktop\\4BWinter2022\\MSCI598\\msci-nlp-w22\\a2\\data\\someSentences.txt mnb_uni