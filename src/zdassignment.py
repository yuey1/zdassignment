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
    if hi - lo + 1 < n:
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
        rank_selection(input_list,sort_key,pivot_idx,hi,n)
if __name__ == "__main__":
    print(count_doc_number(sys.argv[1]))
