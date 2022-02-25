import sys
import time


def run(Path_To_txt, classifier_type):

    return 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"only {len(sys.argv)} arguments provided. Please enter 'py .\a4\inference.py C:\\Path\\to\\afile.txt classifier_to_use'\nSESSION TERMINATED.")
        sys.exit(0)
    else:
        start_time = time.time()
        run(sys.argv[1], sys.argv[2])
        print("\nCOMPLETED IN --- %s seconds ---" % (time.time() - start_time))
# py .\a4\inference.py C:\\Users\\Ruhol\\Desktop\\4BWinter2022\\MSCI598\\msci-nlp-w22\\a4\\data\\testInference.txt relu
