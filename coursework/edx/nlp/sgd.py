import os, csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier

train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/asnlib/public/imdb_te.csv" # test data for grade evaluation

'''Implement this module to extract
and combine text files under train_path directory into 
imdb_tr.csv. Each text file in train_path should be stored 
as a row in imdb_tr.csv. And imdb_tr.csv should have two 
columns, "text" and label'''
def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    os.remove(outpath + name)
    fout = open(outpath + name,'a')
    writer = csv.writer(fout)
    writer.writerow(['row_number','text','polarity'])
    
    cnt = 1
    for filename in os.listdir(inpath + 'neg/'):
        f = open(inpath + 'neg/' + filename)
        writer.writerow([cnt,f.read(),0]) #'\n' + str(cnt) + ',' + f.read() + ',0')
        f.close()
        cnt += 1
      
    for filename in os.listdir(inpath + 'pos/'):
        f = open(inpath + 'pos/' + filename)
        writer.writerow([cnt,f.read(),1]) #('\n' + str(cnt) + ',' + f.read() + ',1')
        f.close()
        cnt += 1

    fout.close()
    pass
  
if __name__ == "__main__":
    # imdb_data_preprocess(train_path)
    stop_words_set = open("./stopwords.en.txt", "r").read().split('\n') #pd.read_csv("./stopwords.en.txt",header=None)

    train_set = pd.read_csv("./imdb_tr.csv")
    test_set = pd.read_csv("./imdb_te.csv",encoding = "ISO-8859-1")
    # test_set = pd.read_csv(test_path,encoding = "ISO-8859-1")

    corpus = train_set['text']
    y_train = train_set['polarity']
    test = test_set['text']

    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    unigramVectorizer = CountVectorizer(stop_words = stop_words_set)
    X_train_unigram = unigramVectorizer.fit_transform(corpus)
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(X_train_unigram, y_train)

    X_test_unigram = unigramVectorizer.transform(test)
    y_predict = clf.predict(X_test_unigram)
    y_merged = '\n'.join(map(str,y_predict))
    with open("unigram.output.txt",'w') as f:
        f.write(y_merged)

    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    bigramVectorizer = CountVectorizer(stop_words = stop_words_set,ngram_range=(1, 2))
    X_train_bigram = bigramVectorizer.fit_transform(corpus)
    clf_b = SGDClassifier(loss="hinge", penalty="l1")
    clf_b.fit(X_train_bigram, y_train)

    X_test_bigram = bigramVectorizer.transform(test)
    y_b_predict = clf_b.predict(X_test_bigram)
    y_b_merged = '\n'.join(map(str,y_b_predict))
    with open("bigram.output.txt",'w') as f:
        f.write(y_b_merged)

    '''train a SGD classifier using unigram representation
    with tf-itrain_set, predict sentiments on imdb_te.csv, and write 
    output to unigram.output.txt'''
    unigramVectorizer = TfidfVectorizer(stop_words = stop_words_set)
    X_train_unigram = unigramVectorizer.fit_transform(corpus)
    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(X_train_unigram, y_train)

    X_test_unigram = unigramVectorizer.transform(test)
    y_predict = clf.predict(X_test_unigram)
    y_merged = '\n'.join(map(str,y_predict))
    with open("unigramtfidf.output.txt",'w') as f:
        f.write(y_merged)

    '''train a SGD classifier using bigram representation
    with tf-itrain_set, predict sentiments on imdb_te.csv, and write 
    output to unigram.output.txt'''
    bigramVectorizer = TfidfVectorizer(stop_words = stop_words_set,ngram_range=(1, 2))
    X_train_bigram = bigramVectorizer.fit_transform(corpus)
    clf_b = SGDClassifier(loss="hinge", penalty="l1")
    clf_b.fit(X_train_bigram, y_train)

    X_test_bigram = bigramVectorizer.transform(test)
    y_b_predict = clf_b.predict(X_test_bigram)
    y_b_merged = '\n'.join(map(str,y_b_predict))
    with open("bigramtfidf.output.txt",'w') as f:
        f.write(y_b_merged)