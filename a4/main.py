import sys
import time


def main(Path_To_data):

    return 1

if __name__=="__main__":
    if len(sys.argv) < 2:
        print(f"only one argument ({sys.argv[0]}) was read. Please enter 'py .\a4\main.py C:\\Path\\to\\Data_Folder'.\nSESSION TERMINATED.")
        sys.exit(0)
    else:
        start_time = time.time()
        main(sys.argv[1])
        print("COMPLETED IN --- %s seconds ---" % (time.time() - start_time))
    # py .\a4\main.py C:\\Users\\Ruhol\\Desktop\\4BWinter2022\\MSCI598\\msci-nlp-w22\\a1\\data