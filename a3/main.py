import sys
import time

# Write a python script using genism library to train a Word2Vec model on the Amazon corpus.
# Output: Word2vec model trained on the entire Amazon corpus (pos.txt + neg.txt)

def main(arg):
    return 1

if __name__=="__main__":
    if len(sys.argv) < 2:
        print(f"only one argument ({sys.argv[0]}) was read. Please enter 'py .\A1\main.py C:\\Path\\to\\Data_Folder'.\nSESSION TERMINATED.")
        sys.exit(0)
    else:
        start_time = time.time()
        main(sys.argv[1])
        print("COMPLETED IN --- %s seconds ---" % (time.time() - start_time))
    # py .\a3\main.py C:\\Users\\Ruhol\\Desktop\\4BWinter2022\\MSCI598\\msci-nlp-w22\\a3\\data