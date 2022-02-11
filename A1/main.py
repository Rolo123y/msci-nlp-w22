import re
import sys
import time
import random as rd
import csv
import os.path

def write_outto_csv(Path, filename, data):
    fullPath = Path + "\\" + filename
    if os.path.isfile(fullPath) == True:
        print(f'\tfile already exists in {fullPath}. Please delete it and try again!')
        return -1
    else:
        with open(fullPath, "w", newline='') as f:
            write = csv.writer(f,quoting=csv.QUOTE_ALL)
            for item in data:
                write.writerow(item)
        f.close()
        print(f'\tCreated file named {filename} located {fullPath}')
    return 1

def tokenize(processed_list_of_Sentences):
    words = []
    for i in range(len(processed_list_of_Sentences)):
        words.append(re.split('\s+', processed_list_of_Sentences[i]))
    return words

def get_stopwords(PathToData):
    stopwords = []
    PathToStopwords = (PathToData + "\\stopwords.txt").strip()
    with open(PathToStopwords) as f:
        for line in f:
            line = re.sub(r'\n', '', line)
            stopwords.append(line)
    f.close()
    return stopwords

def remove_patterns(list_of_Strings):
    list_of_char = ['!', '"', '\-+', '#', '$', '%', '&', '(', ')', '*', '+', '/', ':', ';', '<', '=', '>', '@', '\[',  '\]', '\\\\','^', '`', '{', '|', '}', '~', '\n', '\t', '\'']
    pattern = '[' + ''.join(list_of_char) + ']'
    for i in range(len(list_of_Strings)):
        list_of_Strings[i] = re.sub(pattern, '' , list_of_Strings[i])
        list_of_Strings[i] = re.sub(r' +', ' ' , list_of_Strings[i])
        list_of_Strings[i] = re.sub(r'\.{2,}', ' ' , list_of_Strings[i])
    return list_of_Strings

def remove_Stopwords(tokenized_list_of_Sentences, listOfStopwords):
    lists_of_processed_sentencesTokens_WO_Stopwords = tokenized_list_of_Sentences.copy()
    for i in range(len(lists_of_processed_sentencesTokens_WO_Stopwords)):
        sentence = lists_of_processed_sentencesTokens_WO_Stopwords[i]
        new_Sentence = sentence.copy()
        for j in range(len(sentence)):
            word = sentence[j]
            if word.lower() in listOfStopwords:
                if word in new_Sentence:
                    new_Sentence.remove(word)
                    lists_of_processed_sentencesTokens_WO_Stopwords[i] = new_Sentence
    return lists_of_processed_sentencesTokens_WO_Stopwords

def Iterate_File(PathToData, FilePath, tokenized_Sentences_W_Stopwords, tokenized_Sentences_WO_Stopwords, vocabulary_W_Stopwords, vocabulary_WO_Stopwords, labels):
    with open(FilePath, "r") as f:
        list_of_Sentences = f.readlines()
        list_of_Stopwords = get_stopwords(PathToData)

        # Remove special characters
        processed_list_of_SentencesStrings_W_Stopwords = remove_patterns(list_of_Sentences)

        # Tokenized list of sentences with Stopwords: Each item in the list is a list of tokens
        lists_of_processed_sentencesTokens_W_Stopwords = tokenize(processed_list_of_SentencesStrings_W_Stopwords)

        # Tokenized list of sentences without Stopwords: Each item in the list is a list of tokens
        lists_of_processed_sentencesTokens_WO_Stopwords = remove_Stopwords(lists_of_processed_sentencesTokens_W_Stopwords, list_of_Stopwords)

        # WITH Stopwords: Append to sentence_list and vocab
        for sentence in lists_of_processed_sentencesTokens_W_Stopwords:
            tokenized_Sentences_W_Stopwords.append(sentence)
            labels.append(FilePath[-7:-4])
            for word in sentence:
                if word in vocabulary_W_Stopwords:
                    vocabulary_W_Stopwords[word] += 1
                else:
                    vocabulary_W_Stopwords[word] = 1

         # WOTHOUT Stopwords: Append to sentence_list and vocab
        for sentence in lists_of_processed_sentencesTokens_WO_Stopwords:
            tokenized_Sentences_WO_Stopwords.append(sentence)
            for word in sentence:
                if word in vocabulary_WO_Stopwords:
                    vocabulary_WO_Stopwords[word] += 1
                else:
                    vocabulary_WO_Stopwords[word] = 1
    f.close()

    return [tokenized_Sentences_W_Stopwords, tokenized_Sentences_WO_Stopwords, vocabulary_W_Stopwords, vocabulary_WO_Stopwords, labels]

def main(PathToData):
    vocabulary_W_Stopwords = {}
    vocabulary_WO_Stopwords = {}
    labels = []
    PathToNeg = (PathToData + "\\neg.txt").strip()
    PathToPos = (PathToData + "\\pos.txt").strip()

    print("Processing neg.txt ....")
    Processed_File = Iterate_File(PathToData, PathToNeg, [], [], vocabulary_W_Stopwords, vocabulary_WO_Stopwords, labels)
    tokenized_Sentences_W_Stopwords = Processed_File[0]
    tokenized_Sentences_WO_Stopwords = Processed_File[1]
    vocabulary_W_Stopwords = Processed_File[2]
    vocabulary_WO_Stopwords = Processed_File[3]
    labels = Processed_File[4]
    print(f'SENTENCES WITH STOPWORDS len: {len(tokenized_Sentences_W_Stopwords)}\nSENTENCES WITHOUT STOPWORDS: {len(tokenized_Sentences_WO_Stopwords)}\nVOCAB WITH STOPWORDS len: {len(vocabulary_W_Stopwords)}\nVOCAB WITHOUT STOPWORDS len: {len(vocabulary_WO_Stopwords)}\n')

    print("Processing pos.txt ....")
    Processed_File = Iterate_File(PathToData, PathToPos, tokenized_Sentences_W_Stopwords, tokenized_Sentences_WO_Stopwords, vocabulary_W_Stopwords, vocabulary_WO_Stopwords, labels)
    tokenized_Sentences_W_Stopwords = Processed_File[0]
    tokenized_Sentences_WO_Stopwords = Processed_File[1]
    vocabulary_W_Stopwords = Processed_File[2]
    vocabulary_WO_Stopwords = Processed_File[3]
    labels = Processed_File[4]
    print(f'SENTENCES WITH STOPWORDS len: {len(tokenized_Sentences_W_Stopwords)}\nSENTENCES WITHOUT STOPWORDS: {len(tokenized_Sentences_WO_Stopwords)}\nVOCAB WITH STOPWORDS len: {len(vocabulary_W_Stopwords)}\nVOCAB WITHOUT STOPWORDS len: {len(vocabulary_WO_Stopwords)}')
    

    posCount = 0
    negCount = 0
    for item in labels:
        if item == 'pos':
            posCount += 1
        else:
            negCount += 1
    print(f'labels {len(labels)}\npos {posCount}\nneg {negCount}')

    print("Please wait while the program creates the output files ...")
    OutData_W_Stopwords = {}
    TrainData_W_Stopwords = {}
    ValidData_W_Stopwords = {}
    TestData_W_Stopwords = {}

    OutData_WO_Stopwords = {}
    TrainData_WO_Stopwords = {}
    ValidData_WO_Stopwords = {}
    TestData_WO_Stopwords = {}

    train_ratio = 0.80
    validation_ratio = 0.10 # test is also 10%
    
    rd.seed(233)
    c = list(zip(tokenized_Sentences_W_Stopwords, tokenized_Sentences_WO_Stopwords, labels))
    rd.shuffle(c)

    OutData_W_Stopwords, OutData_WO_Stopwords, fin_lables = zip(*c)

    TrainData_W_Stopwords = OutData_W_Stopwords[:int((len(tokenized_Sentences_W_Stopwords)+1)*train_ratio)]
    ValidData_W_Stopwords = OutData_W_Stopwords[int((len(tokenized_Sentences_W_Stopwords)+1)*train_ratio):int((len(tokenized_Sentences_W_Stopwords)+1)*(train_ratio+validation_ratio))]
    TestData_W_Stopwords = OutData_W_Stopwords[int((len(tokenized_Sentences_W_Stopwords)+1)*(train_ratio+validation_ratio)):]

    TrainData_WO_Stopwords = OutData_WO_Stopwords[:int((len(tokenized_Sentences_WO_Stopwords)+1)*train_ratio)]
    ValidData_WO_Stopwords = OutData_WO_Stopwords[int((len(tokenized_Sentences_WO_Stopwords)+1)*train_ratio):int((len(tokenized_Sentences_WO_Stopwords)+1)*(train_ratio+validation_ratio))]
    TestData_WO_Stopwords = OutData_WO_Stopwords[int((len(tokenized_Sentences_WO_Stopwords)+1)*(train_ratio+validation_ratio)):]

    # print(f'\n{len(OutData_W_Stopwords)}\n{len(TrainData_W_Stopwords)}\n{len(ValidData_W_Stopwords)}\n{len(TestData_W_Stopwords)}\n{len(OutData_WO_Stopwords)}\n{len(TrainData_WO_Stopwords)}\n{len(ValidData_WO_Stopwords)}\n{len(TestData_WO_Stopwords)}\n')

    write_outto_csv(PathToData , "\\out.csv", OutData_W_Stopwords)
    write_outto_csv(PathToData , "\\train.csv", TrainData_W_Stopwords)
    write_outto_csv(PathToData , "\\val.csv", ValidData_W_Stopwords)
    write_outto_csv(PathToData , "\\test.csv", TestData_W_Stopwords)
    write_outto_csv(PathToData , "\\out_ns.csv", OutData_WO_Stopwords)
    write_outto_csv(PathToData , "\\train_ns.csv", TrainData_WO_Stopwords)
    write_outto_csv(PathToData , "\\val_ns.csv", ValidData_WO_Stopwords)
    write_outto_csv(PathToData , "\\test_ns.csv", TestData_WO_Stopwords)

    lablesPath = PathToData + "\\\labels.txt" 
    if os.path.isfile(lablesPath) == True:
        print(f'file already exists in {lablesPath}. Please delete it and try again!')
        return -1
    else:
        with open(lablesPath, "w") as f:
            for i in range(len(fin_lables)):
                if i < (len(fin_lables) - 1):
                    f.write(fin_lables[i] + '\n')
                else:
                    f.write(fin_lables[i])
        f.close()
        print(f'\tCreated file named \labels.txt located {lablesPath}')

if __name__=="__main__":
    if len(sys.argv) < 2:
        print(f"only one argument ({sys.argv[0]}) was read. Please enter 'py .\A1\main.py C:\\Path\\to\\Data_Folder'.\nSESSION TERMINATED.")
        sys.exit(0)
    else:
        start_time = time.time()
        main(sys.argv[1])
        print("COMPLETED IN --- %s seconds ---" % (time.time() - start_time))
    # py .\a1\main.py C:\\Users\\Ruhol\\Desktop\\4BWinter2022\\MSCI598\\msci-nlp-w22\\a1\\data