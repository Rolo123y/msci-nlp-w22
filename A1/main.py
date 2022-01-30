import re
import sys
import time
import random as rd

def write_outto_csv(Path, filename, data):
    fullPath = Path + "\\" + filename
    with open(fullPath, "w") as f:
        f.write(data)
    f.close()
    print(f'Created file named {filename} located {fullPath}')
    return 1

def tokenize(processed_list_of_Sentences):
    words = []
    for i in range(len(processed_list_of_Sentences)):
        words.append(re.split("\s+", processed_list_of_Sentences[i])) 
    return words

def get_stopwords(PathToData):
    stopwords = []
    PathToStopwords = (PathToData + "\\Stopwords.txt").strip()
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

def Iterate_File(PathToData, FilePath, tokenized_Sentences_W_Stopwords, tokenized_Sentences_WO_Stopwords, vocabulary_W_Stopwords, vocabulary_WO_Stopwords):
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

    return [tokenized_Sentences_W_Stopwords, tokenized_Sentences_WO_Stopwords, vocabulary_W_Stopwords, vocabulary_WO_Stopwords]

def main(PathToData):
    vocabulary_W_Stopwords = {}
    vocabulary_WO_Stopwords = {}
    PathToNeg = (PathToData + "\\neg.txt").strip()
    PathToPos = (PathToData + "\\pos.txt").strip()

    print("Processing neg.txt ....")
    Processed_File = Iterate_File(PathToData, PathToNeg, [], [], vocabulary_W_Stopwords, vocabulary_WO_Stopwords)
    tokenized_Sentences_W_Stopwords = Processed_File[0]
    tokenized_Sentences_WO_Stopwords = Processed_File[1]
    vocabulary_W_Stopwords = Processed_File[2]
    vocabulary_WO_Stopwords = Processed_File[3]
    print(f'SENTENCES WITH STOPWORDS len: {len(tokenized_Sentences_W_Stopwords)}\nSENTENCES WITHOUT STOPWORDS: {len(tokenized_Sentences_WO_Stopwords)}\nVOCAB WITH STOPWORDS len: {len(vocabulary_W_Stopwords)}\nVOCAB WITHOUT STOPWORDS len: {len(vocabulary_WO_Stopwords)}\n\n')

    print("Processing pos.txt ....")
    Processed_File = Iterate_File(PathToData, PathToPos, tokenized_Sentences_W_Stopwords, tokenized_Sentences_WO_Stopwords, vocabulary_W_Stopwords, vocabulary_WO_Stopwords)
    tokenized_Sentences_W_Stopwords = Processed_File[0]
    tokenized_Sentences_WO_Stopwords = Processed_File[1]
    vocabulary_W_Stopwords = Processed_File[2]
    vocabulary_WO_Stopwords = Processed_File[3]
    print(f'SENTENCES WITH STOPWORDS len: {len(tokenized_Sentences_W_Stopwords)}\nSENTENCES WITHOUT STOPWORDS: {len(tokenized_Sentences_WO_Stopwords)}\nVOCAB WITH STOPWORDS len: {len(vocabulary_W_Stopwords)}\nVOCAB WITHOUT STOPWORDS len: {len(vocabulary_WO_Stopwords)}')
    
    print("Creating output files ...")
    OutData_W_Stopwords = {}
    TrainData_W_Stopwords = {}
    ValidData_W_Stopwords = {}
    TestData_W_Stopwords = {}
    OutData_WO_Stopwords = {}
    TrainData_WO_Stopwords = {}
    ValidData_WO_Stopwords = {}
    TestData_WO_Stopwords = {}

    # Split the data into train, val, test with and without stopwords and assign to above variables
    # I have a big list of tokenized sentences
    #   - shuffle the sentences using random.shuffle
    #   - randomly grab 80% of the shuffled list as training set
    #   - the other 20% as testing set
    #   - 

    # write_outto_csv(PathToData , "\\out.csv", OutData_W_Stopwords)
    # write_outto_csv(PathToData , "\\train.csv", TrainData_W_Stopwords)
    # write_outto_csv(PathToData , "\\val.csv", ValidData_W_Stopwords)
    # write_outto_csv(PathToData , "\\test.csv", TestData_W_Stopwords)
    # write_outto_csv(PathToData , "\\out_ns.csv", OutData_WO_Stopwords)
    # write_outto_csv(PathToData , "\\train_ns.csv", TrainData_WO_Stopwords)
    # write_outto_csv(PathToData , "\\val_ns.csv", ValidData_WO_Stopwords)
    # write_outto_csv(PathToData , "\\test_ns.csv", TestData_WO_Stopwords)

if __name__=="__main__":
    if len(sys.argv) < 2:
        print(f"only one argument ({sys.argv[0]}) was read. Please enter 'py .\A1\main.py C:\\Path\\to\\Data_Folder'.\nSESSION TERMINATED.")
        sys.exit(0)
    else:
        start_time = time.time()
        main(sys.argv[1])
        print("--- %s seconds ---" % (time.time() - start_time))
    # py .\A1\main.py C:\\Users\\Ruhol\\Desktop\\4BWinter2022\\MSCI598\\msci-nlp-w22\\A1\\data