"""
zd assignment
Yue Yu
yuey1@andrew.cmu.edu
"""
import sys
import os
import re
import nltk
import math
def count_doc_number(path):
    """
    count the number of documents 
    in the directory
    @param path, the directory path
    @return the number of files in that path
    """
    if not os.path.isdir(path):
        return 0
    return len(os.listdir(path))    
def word_tokenizer(word):
    """
    split the string based on 
    space tab . ! ? :, and then get rid of 
    all non alphabetic char in each word
    @param word, the string
    @return a list of words
    """
    word = word.lower() #convert all char into lower case
    word_list = re.compile("[\\s+ || ,.!?]").split(word)
    #remove all non alphabetical char in each word
    for i in xrange(len(word_list)):
        word_list[i] = re.sub("\\W",'',word_list[i])    
    return word_list
def read_files(path):
    """
    read all document from the path into memory
    count the number of occurence of each word and
    store it in a hash table
    @param path, directory which contains all files
    @return a dictionary, the count of each word in the 
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
    word_count = {} #initialized the hash map
    for f in files:
        file_path = path + "/" + f
        with open(file_path,'r') as fid:
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
def rank_selection(input_list,sort_key,lo,hi,n):
    """
    partially sort the list according to some keys using pivoting
    technique, the running time is O(log n)
    @param input_list, a list of tuples, in our case it would be
    (word, count)
    @param sort_key, the index of keys in the tuple we want to sort
    @param lo, define the left bound of the list we want to sort
    @param hi, define the right bount of the list we want to sort
    @param n, the number of entries we want, in our case is 10
    @return none, it sorts the list in place
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
    #base condition
    if n == 0:
        return 
    pivot = input_list[hi][sort_key]
    lo_runner = lo
    hi_runner = hi - 1
    while lo_runner < hi_runner:
        if input_list[lo_runner][sort_key] < pivot:
            lo_runner += 1
        else:
            #swap the lo and hi
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

    target_n = hi - pivot_idx + 1
    if target_n == n:
        return 
    elif target_n < n:
        rank_selection(input_list,sort_key,lo,pivot_idx-1,n-target_n)
    else:
        rank_selection(input_list,sort_key,pivot_idx+1,hi,n)
def get_nonstop_word(word_count,n):
    """
    get the n most frequent non stop word
    @param word_count, word count hash table
    @paran n, number of non stop word we want
    @return a list of target word
    """
    # get rid of all stop word
    tmp_word_list = []
    for t in word_count.keys():
        if not t in nltk.corpus.stopwords.words():
            tmp_word_list.append((t,word_count[t]))
    rank_selection(tmp_word_list,1,0,len(tmp_word_list)-1,n)
    result = []
    for t in tmp_word_list[-n:]:
        result.append(t[0])
    return result
def get_verb(word_count,n):
    """
    get the n most frequent verb for some types 
    of documents
    @param word_count, the word count hash map
    @param n, number of word we want
    @return a list of string
    """
    tags = nltk.pos_tag(word_count.keys())
    verb_list = []
    for t in tags:
        if "V" in t[1]:
            verb_list.append((t[0],word_count[t[0]]))
    rank_selection(verb_list,1,0,len(verb_list)-1,n)
    result = []
    for t in verb_list[-n:]:
        result.append(t[0])
    return result
def get_summary(path):
    """
    print get summary of the current type of document
    including
    1 number of documents
    2 number of unique words
    3 top 10 most frequent non stop words
    4 top 10 most frequent verbs
    @param path, the directory of all documents
    @return none
    """
    if not os.path.isdir(path):
        print "Invalid input path"
        return
    word_count = read_files(path)
    doc_number = count_doc_number(path)
    unique = len(word_count)
    non_stop_words = get_nonstop_word(word_count,10)
    verbs = get_verb(word_count,10)
    print("Number of files: " + str(doc_number) + "\n")
    print("Number of unique words: " + str(unique) + "\n")
    print("10 most frequent non stop words: ")
    for s in non_stop_words:
        print(s)
    print "\n"
    print("10 most frequent verbs: ")
    for s in verbs:
        print(s)
def naive_bayes_train(path1,label1,path2,label2):
    """
    train the naive bayes classifier
    @param path1, directory containing all type 1 document
    @param label1, string, the type1 name
    @param path2, directory containing all type 2 document
    @param label2, string, the type2 name
    @return dictionary, key:label, value: word_count table
    """
    result = {label1:read_files(path1),label2:read_files(path2)}
    return result
def naive_bayes_predict(model,doc):
    """
    predict the type of a document given the model
    @param model, dictionary, the naive bayes classifier
    @param doc, string, the document body
    @return string, the predicted type
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
        typea_score += math.log(float(count_typea+1)/(typea_total+len(model["typea"])))
        typeb_score += math.log(float(count_typeb+1)/(typeb_total+len(model["typeb"])))
    typea_score += math.log(float(typea_total)/(typea_total+typeb_total))
    typeb_score += math.log(float(typeb_total)/(typea_total+typeb_total))
    if typea_score >= typeb_score:
        return "typea"
    else:
        return "typeb"
def naive_bayes_classify(path,model):
    """
    predict the type of every document in the directory
    @param path, directory containing all documents
    @param model, dictionary, the naive bayes model
    """
    if not os.path.isdir(path):
        print "Invalid input path"
        return
    if path[-1] == '/':
        path = path[:-1]
    files = os.listdir(path)
    for f in files:
        file_path = path + "/" + f
        with open(file_path,'r') as fid:
            tmp_doc = fid.read()
        print(f+" type: " + naive_bayes_predict(model,tmp_doc))
if __name__ == "__main__":
    #print(count_doc_number(sys.argv[1]))
    #get_summary(sys.argv[1])
    model = naive_bayes_train(sys.argv[1],sys.argv[2],
    sys.argv[3],sys.argv[4])
    naive_bayes_classify(sys.argv[5],model)
