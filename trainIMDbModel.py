'''
Created on 5 lut 2018

Model to properly train classification of IMDb reviews

@author: mgdak

Credits to:
1. https://github.com/giuseppebonaccorso/Reuters-21578-Classification/blob/master/Text%20Classification.ipynb
2. Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
3. https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py

'''

import argparse
import sys
from pprint import pprint
import numpy as np
import MySQLdb
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2VecKeyedVectors
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM

SQL_CMD_SELECT_ALL = u'select review, pos_neg from imdb.reviews where dtv_classification = %s LIMIT 10000'
MAX_WORDS_NO = 300 #based on histogram data
WORD2VEC_NO_OF_FEATURES = 300 #number of features of a Word2Vec model


def dbConnectionInit(args):
    db = MySQLdb.connect(host=args.dbhost, 
        user=args.dbuser, 
        passwd=args.dbpasw, 
        db=args.dbschema, 
        charset='utf8mb4', 
        init_command='SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci')
    db.set_character_set('utf8mb4')
    dbc = db.cursor()
    dbc.execute('SET CHARACTER SET utf8mb4;')
    dbc.execute('SET character_set_connection=utf8mb4;')
    return dbc, db


def initTokenizers():
    # Load stop-words
    stop_words = set(stopwords.words('english'))
    
    # Initialize tokenizer
    # It's also possible to try with a stemmer or to mix a stemmer and a lemmatizer
    tokenizer = RegexpTokenizer('[\'a-zA-Z]+')
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    return lemmatizer, tokenizer, stop_words

'''
Reads the input data from DB and Tokenizes & Lemmitizes sentences
'''
def prepareDataSet(args, lemmatizer, tokenizer, stop_words, setType):
    X = []
    Y = []
    dbc, db = dbConnectionInit(args)
    dbc.execute(SQL_CMD_SELECT_ALL, (setType, ))
    for review, pos_neg in dbc:
        tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer.tokenize(review) if t.lower() not in stop_words]
        X.append(tokens)
        Y.append(pos_neg)
    return X, Y


def prepareTrainTestDataSet(args):
    lemmatizer, tokenizer, stop_words = initTokenizers()
    X_train, Y_train = prepareDataSet(args, lemmatizer, tokenizer, stop_words, 1)
    X_test, Y_test = prepareDataSet(args, lemmatizer, tokenizer, stop_words, 2)
    
    return X_train, Y_train, X_test, Y_test 


'''
 plots the distribution of senteces lengths after lemmitazation in order to decide on 
 maximum number of words. For the data provided it turned out that 300 covers 98% population
'''
def plotXHist(X_train, X_test):
    X = X_train + X_test
    values = []
    for val in X:
        lenght = len(val)
        values.append(lenght)
    
    plt.hist(values, bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    plt.show()


'''
Builds simple LSTM model as in https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
'''
def crateTrainEvaluateLSTMModel(Y_train, Y_test, X_train_vectorized, X_test_vectorized):
    
    model = Sequential()
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Train...')
    
    # Train model
    model.fit(X_train_vectorized, Y_train, batch_size=32, nb_epoch=15, validation_data=(X_test_vectorized, Y_test))
    
    # Evaluate model
    # TODO: need to change this to cross validation set
    score, acc = model.evaluate(X_test_vectorized, Y_test, batch_size=32)
    print('Score: %1.4f' % score)
    print('Accuracy: %1.4f' % acc)
    model.summary()


def vectorizeInput(X_train, w2v_model, empty_word, missedWords):
    X_train_vectorized = np.zeros(shape=(len(X_train), MAX_WORDS_NO, WORD2VEC_NO_OF_FEATURES), dtype=float)
    
    for idx, document in enumerate(X_train):
        for jdx, word in enumerate(document):
            if jdx == MAX_WORDS_NO:
                break
            else:
                if word in w2v_model:
                    X_train_vectorized[idx, jdx, :] = w2v_model[word]
                else:
                    X_train_vectorized[idx, jdx, :] = empty_word
                    missedWords.append(word)
                    
    return X_train_vectorized


'''
Based on imported word2vec embeddings vectorizes the input
'''
def loadWord2VecAndVectorizeInputs(X_train, X_test, Y_train, word2vecURI):
    w2v_model = Word2VecKeyedVectors.load_word2vec_format(word2vecURI, binary=False)
    WORD2VEC_NO_OF_FEATURES = w2v_model['dog'].shape[0]

    print("num_features = %s", WORD2VEC_NO_OF_FEATURES)
    print("len(X_train) = %s %s", len(X_train), len(Y_train))
    print("len(X_test) = %s", len(X_test))
    
    empty_word = np.zeros(WORD2VEC_NO_OF_FEATURES, dtype=float)
    
    missedWords = []
    
    X_train_vectorized = vectorizeInput(X_train, w2v_model, empty_word, missedWords)

    X_test_vectorized = vectorizeInput(X_test, w2v_model, empty_word, missedWords)

    print("Number of words missing = %s", len(set(missedWords)))
    
    return X_train_vectorized, X_test_vectorized


def main(args):
    #import pydevd;pydevd.settrace();
    pprint(args)
    
    X_train, Y_train, X_test, Y_test = prepareTrainTestDataSet(args)

    #plotXHist(X_train, X_test)
    X_train_vectorized, X_test_vectorized = loadWord2VecAndVectorizeInputs(X_train, X_test, Y_train, args.word2vecmodel)

    if args.networkModel == 'LSTM':
        crateTrainEvaluateLSTMModel(Y_train, Y_test, X_train_vectorized, X_test_vectorized)

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--networkModel', type=str,  choices=['LSTM', 'CNN'],
        help='URI pointing to word 2 vec model to be used'
        , default='LSTM')
    
    parser.add_argument('--word2vecmodel', type=str,
        help='URI pointing to word 2 vec model to be used'
        , default='D:\\tools\\ml_data\\fastText\\crawl-300d-2M.vec')
            
    parser.add_argument('--dbhost', type=str,
        help='MySQL DB host'
        , default='localhost')
    
    parser.add_argument('--dbuser', type=str,
        help='MySQL DB user'
        , default='root')

    parser.add_argument('--dbpasw', type=str,
        help='MySQL DB password'
        , default='')

    parser.add_argument('--dbschema', type=str,
        help='MySQL DB schema name'
        , default='imdb')

    
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))