import sys
import time
import gensim

# Write a python script using genism library to train a Word2Vec model on the Amazon corpus.
# Output: Word2vec model trained on the entire Amazon corpus (pos.txt + neg.txt)

def read_files(Path_To_files):

    with open(Path_To_files+"\\pos.txt", "r") as f:
        pos_content = f.readlines()

    with open(Path_To_files+"\\neg.txt", "r") as f:
        neg_content = f.readlines()
    all_content = (pos_content + neg_content)
    all_content_rev = [gensim.utils.simple_preprocess(sentence) for sentence in all_content]

    return all_content_rev

def main(Path_To_data):

    # Preprocessing sentences
    all_sentences = read_files(Path_To_data)

    # Initialize model
    Word2vec_model = gensim.models.Word2Vec(
        window=10,
        min_count=2,
        workers=4,
    )

    Word2vec_model.build_vocab(all_sentences, progress_per=500)
    print(f'model epochs: {Word2vec_model.epochs}')

    # Training and saving word2vec model
    Word2vec_model.train(all_sentences, total_examples=Word2vec_model.corpus_count, epochs=Word2vec_model.epochs)
    Word2vec_model.save("a3//data//w2v.model")

    # Savings wordvectors
    word_vectors = Word2vec_model.wv
    word_vectors.save("a3//data//word2vec.wordvectors")

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