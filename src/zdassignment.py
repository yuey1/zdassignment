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
        return None
    
    # standarize the path
    if path[-1] == '/':
        path = path[:-1]

    files = os.listdir(path) # get all file names
    word_count = {} #initialized the hash map
    for f in files:
        file_path = path + "/" + f
        with open(file_path,'r') as fid:
            tmp_word = fid.read()
        word_list = word_tokenizer(tmp_word)
        for w in word_list:
            if not w == '':
                if word_count.has_key(w):
                    word_count[w] += 1
                else:
                    word_count[w] = 1
    return word_count
if __name__ == "__main__":
    print(count_doc_number(sys.argv[1]))
