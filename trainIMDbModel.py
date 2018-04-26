'''
Created on 5 lut 2018

Model to properly train classification of IMDb reviews

@author: mgdak

Credits to:
1. https://github.com/giuseppebonaccorso/Reuters-21578-Classification/blob/master/Text%20Classification.ipynb
2. Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

'''

import argparse
import sys
import numpy as np
from pprint import pprint
import MySQLdb
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer

SQL_CMD_SELECT_ALL = u'select review, pos_neg from imdb.reviews where dtv_classification = %s limit 200'


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

def main(args):
    import pydevd;pydevd.settrace();
    pprint(args)
    
    X_train, Y_train, X_test, Y_test = prepareTrainTestDataSet(args)

    maxLen = 0
    X = X_train + X_test
    
    for val in X:
        lenght = len(val)
        if lenght > maxLen:
            maxLen = lenght
    
    pprint(maxLen)

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
            
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