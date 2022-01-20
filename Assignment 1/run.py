from importlib.resources import path


import re

def tokenize(str):
    return re.split("\s+", str.strip())

if __name__=="__main__":
    with open("Data/neg.txt", "r") as f:
        content = f.read(5)
        print(content)

    f.close()