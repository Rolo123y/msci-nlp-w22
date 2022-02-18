import sys
import time

# generates the top-20 most similar words for a given word.

def run(arg):
    return 1

if __name__=="__main__":
    if len(sys.argv) < 3:
        print(f"only {len(sys.argv)} arguments provided. Please enter 'py .\a3\inference.py C:\\Path\\to\\afile.txt\nSESSION TERMINATED.")
        sys.exit(0)
    else:
        start_time = time.time()
        run(sys.argv[1], sys.argv[2])
        print("\nCOMPLETED IN --- %s seconds ---" % (time.time() - start_time))
# py .\a2\inference.py C:\\Users\\Ruhol\\Desktop\\4BWinter2022\\MSCI598\\msci-nlp-w22\\a2\\data\\someSentences.txt mnb_uni