'''
Created on 5 lut 2018

A toolset to process the files from imdb data set to MySQL DB

@author: mgdak

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

'''

import os
import argparse
import sys
from pprint import pprint
import MySQLdb

SQL_COMMAND = u'INSERT INTO reviews (`review`,`source`,`rating`,`source_id`, `pos_neg`) values (%s, %s, %s, %s, %s)'

def extractDataFromFile(args, file):
    sepId = file.find("_")
    reviewId = file[:sepId]
    rating = file[sepId + 1:sepId + 3].replace('.','')
    with open(args.src_dir + file, 'rb') as srcfile:
        content = srcfile.read().decode("UTF-8")

    return reviewId, rating, content


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


def processFiles(args, files):
    dbc, db = dbConnectionInit(args)
    pos_neg = 0
    if args.src_dir.find('pos') >= 0:
        pos_neg = 1
    
    for file in files:
        reviewId, rating, content = extractDataFromFile(args, file)
        pprint(file)
        dbc.execute(SQL_COMMAND, (content, args.src_dir + file, rating, reviewId, pos_neg))
    
    return db

def main(args):
    pprint(args)
    
    files = os.listdir(args.src_dir)
    
    db = processFiles(args, files)
        
    db.commit()
    db.close()

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--src_dir', type=str,
        help='Directory containing txt files to import'
        , default='D:\\tools\\ml_data\\imdb\\script_testing\\')
    
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