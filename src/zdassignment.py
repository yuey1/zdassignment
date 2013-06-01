# zendeal assignment
# Yue Yu
# yuey1@andrew.cmu.edu

import sys
import os
import re
import nltk
import math
import argparse
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV


def count_doc_number(path):
    """count the number of documents 
    in the directory

    Arguments:
    path -- the directory path
    
    return the number of files in that path
    
    """
    if not os.path.isdir(path):
        return 0
    return len(os.listdir(path))    


def word_tokenizer(word):
    """split the string based on 
    space tab . ! ? :, and then get rid of 
    all non alphabetic char in each word
    
    Arguments:
    word -- the word string

    return a list of words

    """
    word = word.lower() # convert all char into lower case
    # remove all non alphabetical char in each word
    word_list = re.compile("[\\s+ || ,.!?]").split(word) 
    for i in xrange(len(word_list)):
        word_list[i] = re.sub("\\W", '', word_list[i])    
    return word_list


def read_files(path):
    """read all document from the path into memory
    count the number of occurence of each word and
    store it in a hash table
    
    Arguments:
    path -- directory which contains all files
    
    return a dictionary, the count of each word in the 
    training set
    
    """
    if not os.path.isdir(path):
        print "Invalid input path"
        return 
    
    # standarize the path
    if path[-1] == '/':
        path = path[:-1]
    total_count = 0 # count the total number of words
    files = os.listdir(path) # get all file names
    word_count = {} # initialized the hash map

    for f in files:
        file_path = path + "/" + f
        with open(file_path, 'r') as fid:
            tmp_word = fid.read()
        word_list = word_tokenizer(tmp_word)
        for w in word_list:
            if not w == '':
                total_count += 1
                if word_count.has_key(w):
                    word_count[w] += 1
                else:
                    word_count[w] = 1
    word_count["#total_count"] = total_count
    return word_count


def rank_selection(input_list, sort_key, lo, hi, n):
    """partially sort the list according to some keys using pivoting
    technique, the running time is O(log n)

    Arguments:
    input_list -- a list of tuples, in our case it would be
    (word, count)
    sort_key -- the index of keys in the tuple we want to sort
    lo -- define the left bound of the list we want to sort
    hi -- define the right bount of the list we want to sort
    n -- the number of entries we want, in our case is 10

    return none, it sorts the list in place

    """
    # input integrity checking
    if len(input_list) == 0:
        return
    if sort_key < 0 or sort_key >= len(input_list[0]):
        print "invalid sort_key"
        return 
    if hi - lo + 1 <= n:
        return 
    if lo < 0 or hi >= len(input_list):
        return 
    if lo >= hi:
        return

    # base condition
    if n == 0:
        return 

    pivot = input_list[hi][sort_key]
    lo_runner = lo
    hi_runner = hi - 1
    while lo_runner < hi_runner:
        if input_list[lo_runner][sort_key] < pivot:
            lo_runner += 1
        else:
            # swap the lo and hi
            tmp_tuple = input_list[lo_runner]
            input_list[lo_runner] = input_list[hi_runner]
            input_list[hi_runner] = tmp_tuple
            hi_runner -= 1
    # at this point, all tuples with keys which are greater than
    # or equal to pivot will be at the right hand side, then swap 
    # the pivot and go to the recursive step
    pivot_idx = lo_runner # lo_runner = hi_runner
    if input_list[lo_runner][sort_key] < pivot:
        pivot_idx = lo_runner + 1
    tmp_tuple = input_list[hi]
    input_list[hi] = input_list[pivot_idx]
    input_list[pivot_idx] = tmp_tuple
    # recursion step
    target_n = hi - pivot_idx + 1
    if target_n == n:
        return 
    elif target_n < n:
        rank_selection(input_list, sort_key, lo, pivot_idx-1, n-target_n)
    else:
        rank_selection(input_list, sort_key, pivot_idx+1, hi, n)


def get_nonstop_word(word_count, n):
    """get the n most frequent non stop word

    Arguments:
    word_count -- word count hash table
    n -- number of non stop word we want
    
    return a list of target word
    
    """
    # get rid of all stop word
    tmp_word_list = []
    for t in word_count.keys():
        if not t in nltk.corpus.stopwords.words() and t[0] != "#":
            tmp_word_list.append((t, word_count[t]))
    rank_selection(tmp_word_list,1,0,len(tmp_word_list)-1, n)
    result = []
    for t in tmp_word_list[-n:]:
        result.append(t[0])
    return result


def get_verb(word_count, n):
    """get the n most frequent verb for some types 
    of documents

    Arguments:
    word_count -- the word count hash map
    n -- number of word we want
    
    return a list of string
    
    """
    tags = nltk.pos_tag(word_count.keys()) # get tags
    verb_list = []
    for t in tags:
        # get verbs
        if "V" in t[1] and t[0][0] != '#':
            verb_list.append((t[0], word_count[t[0]]))
    rank_selection(verb_list, 1, 0, len(verb_list)-1, n) # select top n words
    result = []
    for t in verb_list[-n:]:
        result.append(t[0])
    return result


def get_summary(path):
    """print get summary of the current type of document
    including
    1 number of documents
    2 number of unique words
    3 top 10 most frequent non stop words
    4 top 10 most frequent verbs

    Arguments:
    path -- the directory of all documents
    
    """
    if not os.path.isdir(path):
        print "Invalid input path"
        return
    word_count = read_files(path)
    doc_number = count_doc_number(path)
    unique = len(word_count)
    non_stop_words = get_nonstop_word(word_count, 10)
    verbs = get_verb(word_count, 10)
    print("Number of files: " + str(doc_number) + "\n")
    print("Number of unique words: " + str(unique) + "\n")
    print("10 most frequent non stop words: ")
    for s in non_stop_words:
        print(s)
    print "\n"
    print("10 most frequent verbs: ")
    for s in verbs:
        print(s)


def naive_bayes_train(path1, label1, path2, label2):
    """train the naive bayes classifier
    
    Arguments:
    path1 -- directory containing all type 1 document
    label1 -- string, the type1 name
    path2 -- directory containing all type 2 document
    label2 -- string, the type2 name

    return dictionary, key:label, value: word_count table
    
    """
    result = {label1:read_files(path1), label2:read_files(path2)}
    return result


def naive_bayes_predict(model, doc):
    """ predict the type of a document given the model

    Arguments:

    model -- dictionary, the naive bayes classifier
    doc -- string, the document body

    return int, the predicted type, typea is 1, typeb is 0
    
    """
    word_list = word_tokenizer(doc)
    # compute the score for each type
    typea_score = 0
    typea_total = model["typea"]["#total_count"]
    typeb_score = 0
    typeb_total = model["typeb"]["#total_count"]
    for w in word_list:
        count_typea = 0
        count_typeb = 0
        if model["typea"].has_key(w):
            count_typea = model["typea"][w]
        if model["typeb"].has_key(w):
            count_typeb = model["typeb"][w]
        # use plus 1 smooth term
        typea_score += math.log(float(count_typea+1) / (typea_total+len(model["typea"])))
        typeb_score += math.log(float(count_typeb+1) / (typeb_total+len(model["typeb"])))
    typea_score += math.log(float(typea_total) / (typea_total+typeb_total))
    typeb_score += math.log(float(typeb_total) / (typea_total+typeb_total))
    if typea_score >= typeb_score:
        return 1
    else:
        return 0


def naive_bayes_classify(path, model):
    """predict the type of every document in the directory

    Arguments:
    path -- directory containing all documents
    model -- dictionary, the naive bayes model
    
    return a list of labels
    """
    predict_label = []
    if not os.path.isdir(path):
        print "Invalid input path"
        return
    if path[-1] == '/':
        path = path[:-1]
    files = os.listdir(path)
    for f in files:
        file_path = path + "/" + f
        with open(file_path, 'r') as fid:
            tmp_doc = fid.read()
        predict_label.append(naive_bayes_predict(model, tmp_doc))
    return predict_label


def classifiers(train_data_path, test_data_path):
    """train and test SVM, Naive Bayes, Decision tree
    classifiers with different feature selection methods by 
    using scikit library
     
    Arguments:
    train_data_path -- director containing training data, in this case
                  it contains typea and typeb folder
    test_data_path -- director containing test data, in this case
                 it contains typea and typeb folder
    
    return model evaluation reports for each classifier 

    """
    # remove the last possible / in the path
    if train_data_path[-1] == '/':
        train_data_path = train_data_path[:-1]
    if test_data_path[-1] == '/':
        test_data_path = test_data_path[:-1]
    # check input integrity
    if not os.path.isdir(train_data_path) or not os.path.isdir(test_data_path):
        print 'Invalid Input directory'
        return
    labels = os.listdir(train_data_path)
    if len(labels) != 2:
        print 'there must be two different type of documents'
        return
    # get the labels
    type1 = labels[0]
    type2 = labels[1]
    if (not type1 in os.listdir(test_data_path) or
       not type2 in os.listdir(test_data_path)):
        print 'Wrong test files'
        return

    # first read all data into memory
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    # read training data and generate labels
    tmp_path = train_data_path + '/' + type1
    tmp_files = os.listdir(tmp_path)
    for f in tmp_files:
        with open(tmp_path+'/'+f, 'r') as fid:
            train_data.append(fid.read())
            train_label.append(0) # type1 is 0 type2 is 1
    tmp_path = train_data_path + '/' + type2
    tmp_files = os.listdir(tmp_path)
    for f in tmp_files:
        with open(tmp_path+'/'+f, 'r') as fid: 
            train_data.append(fid.read())
            train_label.append(1)
    # read test data and generate labels
    tmp_path = test_data_path + '/' + type1
    tmp_files = os.listdir(tmp_path)
    for f in tmp_files:
        with open(tmp_path+'/'+f, 'r') as fid:
            test_data.append(fid.read())
            test_label.append(0)
    tmp_path = test_data_path + '/' + type2
    tmp_files = os.listdir(tmp_path)
    for f in tmp_files:
        with open(tmp_path+'/'+f, 'r') as fid:
            test_data.append(fid.read())
            test_label.append(1)
    # use bag of words model to convert all documents into numerical vectors.
    # three methods are used here, 
    # 1 the unigram count, 
    # 2 unigram and bigram count,
    # 3 unigram count with tfidf reweighting
    # besides, all words in training data with frequency less than 0.0005 are ignored
    unigram_vectorizer = CountVectorizer(min_df=0.0005)
    mix_vectorizer = CountVectorizer(ngram_range=(1,2), min_df=0.0005)
    tfidf_transformer = TfidfTransformer()
    # next, for each feature dictionary, build and test SVM, Naive Bayes and 
    # Decision Tree classifiers

    # unigram count
    X = unigram_vectorizer.fit_transform(train_data)
    test_X = unigram_vectorizer.transform(test_data)
    # Naive Bayes
    nb = MultinomialNB()
    y_pred = nb.fit(X, train_label).predict(test_X)
    print 'Naive Bayes with unigram count'
    print(classification_report(test_label, y_pred.tolist(), target_names=labels))
    # Decision Tree
    DT = tree.DecisionTreeClassifier()
    y_pred = DT.fit(X.toarray(), train_label).predict(test_X.toarray())
    print 'DecisionTree with unigram count'
    print(classification_report(test_label, y_pred.tolist(), target_names=labels))
    # Support Vector Machine
    # first use grid search to determine C and gamma
    C_range = 10.0 ** np.arange(-1, 1)
    gamma_range = 10.0 ** np.arange(-1, 1)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(y=np.array(train_label), n_folds=3)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, np.array(train_label))
    SVM = grid.best_estimator_ # the classifier with best C and Gamma.
    y_pred = SVM.fit(X, train_label).predict(test_X)
    print 'SVM with unigram count'
    print(classification_report(test_label, y_pred.tolist(), target_names=labels))

    # unigram and bigram count
    X = mix_vectorizer.fit_transform(train_data)
    test_X = mix_vectorizer.transform(test_data)
    # Naive Bayes
    nb = MultinomialNB()
    y_pred = nb.fit(X, train_label).predict(test_X)
    print 'Naive Bayes with unigram and bigram count'
    print(classification_report(test_label, y_pred.tolist(), target_names=labels))
    # Decision Tree
    DT = tree.DecisionTreeClassifier()
    y_pred = DT.fit(X.toarray(), train_label).predict(test_X.toarray())
    print 'DecisionTree with unigram and bigram count'
    print(classification_report(test_label, y_pred.tolist(), target_names=labels))
    # Support Vector Machine
    # first use grid search to determine C and gamma
    C_range = 10.0 ** np.arange(-1, 1)
    gamma_range = 10.0 ** np.arange(-1, 1)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(y=np.array(train_label), n_folds=3)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, np.array(train_label))
    SVM = grid.best_estimator_ # the classifier with best C and Gamma.
    y_pred = SVM.fit(X, train_label).predict(test_X)
    print 'SVM with unigram and bigram count'
    print(classification_report(test_label, y_pred.tolist(), target_names=labels))

    # unigram count with tfidf
    X = unigram_vectorizer.fit_transform(train_data)
    X = tfidf_transformer.fit_transform(X)
    test_X = unigram_vectorizer.transform(test_data)
    test_X = tfidf_transformer.transform(test_X)
    # Gaussian Naive Bayes
    nb = GaussianNB()
    y_pred = nb.fit(X.toarray(), train_label).predict(test_X.toarray())
    print 'Naive Bayes with tfidf unigram'
    print(classification_report(test_label, y_pred.tolist(), target_names=labels))
    # Decision Tree
    DT = tree.DecisionTreeClassifier()
    y_pred = DT.fit(X.toarray(), train_label).predict(test_X.toarray())
    print 'DecisionTree with tfidf unigram'
    print(classification_report(test_label, y_pred.tolist(), target_names=labels))
    # Support Vector Machine
    # first use grid search to determine C and gamma
    C_range = 10.0 ** np.arange(-1, 1)
    gamma_range = 10.0 ** np.arange(-1, 1)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(y=np.array(train_label), n_folds=3)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, np.array(train_label))
    SVM = grid.best_estimator_ # the classifier with best C and Gamma.
    y_pred = SVM.fit(X, train_label).predict(test_X)
    print 'SVM with tfidf unigram'
    print(classification_report(test_label, y_pred.tolist(), target_names=labels))


if __name__ == "__main__":
    # set up the argument parser
    parser = argparse.ArgumentParser(description='Parse the arguments.')
    parser.add_argument('-s', '--stat', metavar='path', nargs=1,
                        help='get the statistics of current type of documents.')
    parser.add_argument('-c', '--classification', metavar='path', nargs=2, 
                        help='build Naive Bayes classifier and test it,'+
                        'first path is training data path, second path is'+
                        'test data path.')
    parser.add_argument('-p', '--predictions', metavar='path', nargs=2,
                        help='use scikit library to build SVM Naive Bayes and'+
                        'Decision Tree classifiers, print out all test result.'+
                        'first path is training data path, second path is'+
                        'test data path.')
    args = parser.parse_args() # parse all arguments
    if  args.stat != None:
        get_summary(args.stat[0])

    if  args.classification != None:
        # get the labels first
        train_labels = os.listdir(args.classification[0])
        model = naive_bayes_train(args.classification[0]+'/'+train_labels[0], 
                                  train_labels[0], args.classification[0]+'/'
                                  +train_labels[1], train_labels[1])
        # generate test data true label
        labels = [0] * len(os.listdir(args.classification[1]+'/' 
                           +train_labels[0])) # typeb
        labels += [1] * len(os.listdir(args.classification[1]+'/'
                            +train_labels[1])) # typea
        # predict the labels                    
        predict_labels = naive_bayes_classify(args.classification[1] + '/'
        + train_labels[0], model)
        predict_labels += naive_bayes_classify(args.classification[1] + '/'
        + train_labels[1], model)
        # generate the metrics
        print(classification_report(labels, predict_labels, target_names=train_labels))

    if args.predictions != None:
        classifiers(args.predictions[0], args.predictions[1])
