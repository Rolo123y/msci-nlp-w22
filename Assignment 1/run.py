from importlib.resources import path
import re

def tokenize(str):
    str = str.strip()
    return re.split("\s+", str)

def remove_patterns(str):
    pattern = r'[!"#$%&()*+/:;<=>@//\\^`{|}~\\s\\t..+]'
    return re.sub(pattern,r' ', str)

def get_stopwords():
    stopwords = []
    with open("Data/Stopwords.txt") as f:
        for line in f:
            line = re.sub(r'\n',r' ', line)
            stopwords.append(line)
    f.close()
    return stopwords

if __name__=="__main__":
    tokenized_file = []
    word_count = {}
    with open("Data/neg.txt", "r") as f:
        content = f.read()
        processed_line = remove_patterns(content)
        words = tokenize(processed_line)
        tokenized_file.append(words)
        for word in words:
            if word in word_count:
                word_count[word] += 1 
            else:
                word_count[word] = 1 
    f.close()

    # A list of stopwords read from Data/Stopwords.txt
    stopwords = get_stopwords()

    # for k ,v in word_count.items():
        # print(f'{k} -> {v}')

    # print(len(word_count.keys())