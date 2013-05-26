"""
zd assignment
Yue Yu
yuey1@andrew.cmu.edu
"""
import sys
import os
import re
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
def naive_bayes_train(path):
    """
    train the naive bayes classifer
    it's basically tokenizing the documents and
    store the count of each word into a hashtable
    @param path, directory which contains all files
    @return a dictionary, the count of each word in the 
    training set
    """



if __name__ == "__main__":
    print(count_doc_number(sys.argv[1]))
