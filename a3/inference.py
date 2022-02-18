import sys
import time
from gensim.models import KeyedVectors

# generates the top-20 most similar words for a given word.

def read_file(Path_To_txt):

    with open(Path_To_txt, 'r') as f:
        content = f.readlines()
    all_content_rev = [word.strip() for word in content]

    return all_content_rev

def run(Path_To_txt):

    all_words = read_file(Path_To_txt)

    wv = KeyedVectors.load("a3//data//word2vec.wordvectors", mmap='r')

    for word in all_words:
        try:
            print(f'{word} =>\t{wv.most_similar(positive=word, topn=20)}\n\n')
        except:
            print(word, '=> no similar words found.\n\n')

    return 1

if __name__=="__main__":
    if len(sys.argv) < 2:
        print(f"only {len(sys.argv)} arguments provided. Please enter 'py .\a3\inference.py C:\\Path\\to\\afile.txt\nSESSION TERMINATED.")
        sys.exit(0)
    else:
        start_time = time.time()
        run(sys.argv[1])
        print("\nCOMPLETED IN --- %s seconds ---" % (time.time() - start_time))
# py .\a3\inference.py C:\\Users\\Ruhol\\Desktop\\4BWinter2022\\MSCI598\\msci-nlp-w22\\a3\\data\\testInference.txt