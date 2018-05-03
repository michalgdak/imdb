'''
Created on 5 lut 2018

Model to properly train classification of IMDb reviews

@author: mgdak

Credits to:
1. https://github.com/giuseppebonaccorso/Reuters-21578-Classification/blob/master/Text%20Classification.ipynb
2. Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
3. https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
4. https://github.com/bhaveshoswal/CNN-text-classification-keras

'''

import argparse
import sys
from pprint import pprint
import numpy as np
import MySQLdb
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2VecKeyedVectors
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, MaxPool2D, Conv2D, Flatten, Reshape, Dropout, Input, Concatenate, Embedding
from keras import callbacks
import pickle

SQL_CMD_SELECT_ALL = u'select review, pos_neg from imdb.reviews where dtv_classification = %s'
SQL_CMD_SELECT_LIMIT = u'select review, pos_neg from imdb.reviews where dtv_classification = %s LIMIT 100'
MAX_WORDS_NO = 300 #based on histogram data
WORD2VEC_NO_OF_FEATURES = 300 #number of features of a Word2Vec model
SERIALIZE_DATA_FILE_NAME = 'imdb_train_test.p'
FILTER_SIZES = [3,4]
NUM_FILTERS = 128

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
def prepareDataSet(args, lemmatizer, tokenizer, stop_words, setType, mode):
    X = []
    Y = []
    dbc, db = dbConnectionInit(args)
    dbc.execute((SQL_CMD_SELECT_ALL if mode == 'dump' else SQL_CMD_SELECT_LIMIT), (setType, ))
    for review, pos_neg in dbc:
        tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer.tokenize(review) if t.lower() not in stop_words]
        X.append(tokens)
        Y.append(pos_neg)
    return X, Y


def prepareTrainTestValDataSet(args, mode):
    lemmatizer, tokenizer, stop_words = initTokenizers()
    X_train, Y_train = prepareDataSet(args, lemmatizer, tokenizer, stop_words, 1, mode)
    X_test, Y_test = prepareDataSet(args, lemmatizer, tokenizer, stop_words, 2, mode)
    X_val, Y_val = prepareDataSet(args, lemmatizer, tokenizer, stop_words, 3, mode)
    
    return X_train, Y_train, X_test, Y_test, X_val, Y_val 


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
Serializes the trained model
'''
def serializeModel(model, fileName):
    # serialize model to JSON
    model_json = model.to_json()
    with open(fileName + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(fileName + ".h5")
    print("Saved model to disk")
    
    
'''
Builds simple LSTM model as in https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
'''

def createSimpleLSTMModel():
    model = Sequential()
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(MAX_WORDS_NO, WORD2VEC_NO_OF_FEATURES, )))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
    return model, earlystop

'''
This is a workaround for a known bug in gensim get_keras_embedding
'''
def createKerasEmbeddingLayer(w2v_model, word2index):
    vocab_len = len(word2index) + 1
    emb_matrix = np.zeros((vocab_len, WORD2VEC_NO_OF_FEATURES))
    
    for word, index in word2index.items():
        emb_matrix[index, :] = w2v_model[word]

    embedding_layer = Embedding(vocab_len, WORD2VEC_NO_OF_FEATURES, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

'''
Builds simple LSTM model woth Keras Embedding lazer
'''
def createSimpleLSTMWithEmbeddingModel(w2v_model, word2index):
    model = Sequential()
    model.add(createKerasEmbeddingLayer(w2v_model, word2index))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
    return model, earlystop


def crateTrainEvaluateLSTMModel(Y_train, Y_test, Y_val, X_train_vectorized, X_test_vectorized, X_val_vectorized, savedModelName, noOfEpochs, model, w2v_model, word2index):
   
    if model == 'LSTM': 
        model, earlystop = createSimpleLSTMModel()
    else:
        model, earlystop = createSimpleLSTMWithEmbeddingModel(w2v_model,word2index)

    # Train model
    print('Train...')
    history = model.fit(X_train_vectorized, 
                        Y_train, 
                        batch_size=32, 
                        epochs=noOfEpochs, 
                        validation_data=(X_test_vectorized, Y_test),
                        callbacks=[earlystop])
    
    # Evaluate model
    score, acc = model.evaluate(X_val_vectorized, Y_val, batch_size=32)
    print('Score: %1.4f' % score)
    print('Accuracy: %1.4f' % acc)
    model.summary()
    
    plt.plot(history.history['val_acc'], 'r')
    plt.plot(history.history['acc'], 'b')
    plt.title('Performance of model LSTM')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs No')
    plt.savefig(savedModelName + '_initialModel_plot.png')
    serializeModel(model, savedModelName + "_initialModel")


'''
Builds simple LSTM model as in https://github.com/bhaveshoswal/CNN-text-classification-keras
'''
def createCNNModel(w2v_model, word2index):
    inputs = Input(shape=(MAX_WORDS_NO, ), dtype='int32')
    embedding_layer = createKerasEmbeddingLayer(w2v_model, word2index)
    embedding = embedding_layer(inputs)
    reshape = Reshape((MAX_WORDS_NO, WORD2VEC_NO_OF_FEATURES, 1))(embedding)
    
    conv_0 = Conv2D(NUM_FILTERS, kernel_size=(FILTER_SIZES[0], WORD2VEC_NO_OF_FEATURES), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(NUM_FILTERS, kernel_size=(FILTER_SIZES[1], WORD2VEC_NO_OF_FEATURES), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    
    maxpool_0 = MaxPool2D(pool_size=(MAX_WORDS_NO - FILTER_SIZES[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(MAX_WORDS_NO - FILTER_SIZES[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(0.5)(flatten)
    output = Dense(units=1, activation='softmax')(dropout)
    
    # this creates a model
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
    
    return model, earlystop

def crateTrainEvaluateCNNModel(Y_train, Y_test, Y_val, X_train_vectorized, X_test_vectorized, X_val_vectorized, savedModelName, noOfEpochs, w2v_model, word2index):
    
    model, earlystop = createCNNModel(w2v_model, word2index)
    
    # Train model
    print('Train...')
    history = model.fit(X_train_vectorized, 
                        Y_train, 
                        batch_size=32, 
                        epochs=noOfEpochs, 
                        validation_data=(X_test_vectorized, Y_test),
                        callbacks=[earlystop])
    
    # Evaluate model
    score, acc = model.evaluate(X_val_vectorized, Y_val, batch_size=32)
    print('Score: %1.4f' % score)
    print('Accuracy: %1.4f' % acc)
    model.summary()
    
    plt.plot(history.history['val_acc'], 'r')
    plt.plot(history.history['acc'], 'b')
    plt.title('Performance of model LSTM')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs No')
    plt.savefig(savedModelName + '_initialModel_plot.png')
    serializeModel(model, savedModelName + "_initialModel")

    
#TODO: refactor needed    
def vectorizeInput(X_train, w2v_model, empty_word, missedWords, networkModel, word2index):
    if networkModel =='LSTM':
        #For a usual LSTM network the input has to take into account the word embeddings
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
    
    else: 
        #Keras Embedding layer requires id of a word2vec not the embedding 
        X_train_vectorized = np.zeros(shape=(len(X_train), MAX_WORDS_NO), dtype=int)
        
        word2indexId = 1
        
        for idx, document in enumerate(X_train):
            for jdx, word in enumerate(document):
                if jdx == MAX_WORDS_NO:
                    break
                else:
                    if word in w2v_model:
                        if word not in word2index:
                            word2index[word] = word2indexId
                            word2indexId += 1
                        X_train_vectorized[idx, jdx] = word2index[word] 
                    else:
                        X_train_vectorized[idx, jdx] = 0
                        missedWords.append(word)
                    
    return X_train_vectorized


'''
Based on imported word2vec embeddings vectorizes the input
'''
def loadWord2VecAndVectorizeInputs(X_train, X_test, X_val, Y_train, word2vecURI, networkModel):
    
    #load Word2Vec model
    w2v_model = Word2VecKeyedVectors.load_word2vec_format(word2vecURI, binary=False)
    print("vocab_size = %s", len(w2v_model.vocab))
    
    #determine number of features for each word in the model
    WORD2VEC_NO_OF_FEATURES = w2v_model['dog'].shape[0]

    print("num_features = ", WORD2VEC_NO_OF_FEATURES)
    print("len(X_train) = ", len(X_train))
    print("len(Y_train) = ", len(Y_train))
    print("len(X_test) = ", len(X_test))
    
    #define the missing word vector
    empty_word = np.zeros(WORD2VEC_NO_OF_FEATURES, dtype=float)
    
    #create the list to get the all words which we are missing in the Word2Vec model
    missedWords = []
    word2index = {}
    
    #vectorize each input
    X_train_vectorized = vectorizeInput(X_train, w2v_model, empty_word, missedWords, networkModel, word2index)
    X_test_vectorized = vectorizeInput(X_test, w2v_model, empty_word, missedWords, networkModel, word2index)
    X_val_vectorized = vectorizeInput(X_val, w2v_model, empty_word, missedWords, networkModel, word2index)

    print("Number of used words = ", len(set(missedWords)))
    print("Number of words missing = ", len(set(missedWords)))
    
    return X_train_vectorized, X_test_vectorized, X_val_vectorized, w2v_model, word2index


def main(args):
    #import pydevd;pydevd.settrace();
    pprint(args)
    
    if args.mode == 'local':
        #In local mode data is fetched from MySQL where it's been put using processData
        X_train, Y_train, X_test, Y_test, X_val, Y_val = prepareTrainTestValDataSet(args, args.mode)
        #plotXHist(X_train, X_test)     # use this to define MAX_WORDS_NO which will account for most of the data
    elif args.mode == 'dump':
        #dump mode serializes data from MySQL after lemmitization and tokenization
        #this trick makes it easy to send the data to AWS
        X_train, Y_train, X_test, Y_test, X_val, Y_val = prepareTrainTestValDataSet(args, args.mode)
        #plotXHist(X_train, X_test)    
        pickle.dump( [X_train, Y_train, X_test, Y_test, X_val, Y_val], open(SERIALIZE_DATA_FILE_NAME, "wb" ))
        return
    elif args.mode == 'readAndRun':
        #reads the data from serialized file using 'dump' mode
        X_train, Y_train, X_test, Y_test, X_val, Y_val = pickle.loads(open(SERIALIZE_DATA_FILE_NAME, "rb").read())
            
    #vectorize the sentences using Word2Vec from fastText pointed by args.word2vecmodel
    X_train_vectorized, X_test_vectorized, X_val_vectorized, w2v_model, word2index = loadWord2VecAndVectorizeInputs(X_train, X_test, X_val, Y_train, args.word2vecmodel, args.networkModel)

    if args.networkModel == 'CNN':
        #creates and trains model using CNN architecture
        crateTrainEvaluateCNNModel(Y_train, Y_test, Y_val, X_train_vectorized, X_test_vectorized, X_val_vectorized, args.savedModelName, args.no_of_epochs, w2v_model, word2index)
    else:
        #creates and trains model using RNN (LSTM) architecture
        crateTrainEvaluateLSTMModel(Y_train, Y_test, Y_val, X_train_vectorized, X_test_vectorized, X_val_vectorized, args.savedModelName, args.no_of_epochs, args.networkModel, w2v_model, word2index)

    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str,  choices=['local', 'dump', 'readAndRun'],
        help='local - uses MySQL to fetch the data and train selected model, dump - serializes train data sets for AWS usage, readAndRun - reads serailized data and trains selected model'
        , default='local')

    parser.add_argument('--networkModel', type=str,  choices=['LSTM', 'CNN', 'LSTMWithEmbedding'],
        help='LSTM - simple LSTM model using 3d input, CNN - simple CNN, LSTMWithEmbedding - simple LSTM using 2D input with Keras Embedding'
        , default='LSTM')
    
    parser.add_argument('--word2vecmodel', type=str,
        help='URI pointing to word 2 vec model to be used'
        , default='D:\\tools\\ml_data\\fastText\\crawl-300d-2M.vec')
    
    parser.add_argument('--savedModelName', type=str,
        help='File name for the train model persistance'
        , default='LSTM')
    
    parser.add_argument('--no_of_epochs', type=int,
        help='Number of epochs to run.', default=15)
                
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